import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from sklearn import preprocessing
import numpy as np
import dill
import os
import sys
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
from operator import itemgetter

class GenPropData(Dataset):
    def __init__(self, length, data_in, row_size):
        self.length = length
        self.dataframe = data_in
        self.row_size = row_size
        self.pop_order = np.sort(pd.unique(data_in["pop"]))
        self.pop_num = len(self.pop_order)
        self.snp_length = len(data_in["alleles"][0])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Get a random subset of the training df
        glob = self.dataframe
        for_conv = self.dataframe.sample(n=self.row_size)
        arr = np.stack(np.array(for_conv["alleles"]))
        indices = np.lexsort(arr.T[::-1])
        sorted_arr = arr[indices]
        row_order = for_conv.index[indices]
        pop_num = self.pop_num
        # get props for each set
        props = []
        for i in range(pop_num):
          props.append(for_conv["pop"].to_list().count(self.pop_order[i]))
        props = np.array(props)/self.row_size

        x = torch.from_numpy(sorted_arr).float().unsqueeze(0) # Random input vector
        y = torch.from_numpy(props).float() # Label is the sum of elements
        return x, y

class CNNRegressor(nn.Module):
    def __init__(self, row_size, snps, kernal_height, kernal_width, out_channels, h_mpool, w_mpool, pop_num, pooling=1):
        super(CNNRegressor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=(kernal_height,kernal_width), padding=0), # 1 input channel (only one snp layer)
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((h_mpool, w_mpool))
            #nn.MaxPool2d(pooling)
        )
        #self.height_end = (((row_size - kernal_height + 1) - pooling ) // pooling) + 1
        #self.width_end = (((snps - kernal_width + 1) - pooling ) // pooling) + 1
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(out_channels * h_mpool * w_mpool),
            #nn.Linear(out_channels * self.height_end * self.width_end,64), 
            nn.Linear(out_channels * h_mpool * w_mpool,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, pop_num), # output number of variables needed for row
            nn.Softmax(dim=1) # standardize so response totals 1
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

def train_loop(dataloader, model, loss_fn, optimizer, batch_size):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            correct = torch.mean(torch.nn.functional.pairwise_distance(pred,y, p=2))

    test_loss /= num_batches
    correct /= size

def _generate_train_inputs(data_obj, valid_size, cv_splits, cv_reps, seed=123, bootstrap=False):

    if cv_splits == 1:
        train_input, valid_input = data_obj.split_train_test(
            data_obj.train, test_size=valid_size, seed=seed, bootstrap=bootstrap)
        inputs = [(train_input, valid_input)]

    elif cv_splits > 1:
        inputs = data_obj.split_kfcv(
            data_obj.train, n_splits=cv_splits, n_reps=cv_reps, seed=seed, bootstrap=bootstrap)

    return inputs

def _split_input_classifier(clf, input):
        
    train_input, valid_input = input

    X_train = train_input["alleles"]
    X_valid = valid_input["alleles"]
    y_train = train_input["pop"] # one hot encode
    y_valid = valid_input["pop"] # one hot encode

    # Label encode y values
    # clf.label_enc = preprocessing.LabelEncoder()
    y_train = clf.label_enc.fit_transform(y_train)
    y_valid = clf.label_enc.transform(y_valid)

    X_train, y_train = _data_converter(X_train, y_train)
    X_valid, y_valid = _data_converter(X_valid, y_valid)

    return X_train, y_train, X_valid, y_valid

def _split_input_regressor(input):
        
    train_input, valid_input = input

    X_train = train_input["alleles"]
    X_valid = valid_input["alleles"]
    y_train = train_input[["x", "y"]]
    y_valid = valid_input[["x", "y"]]

    X_train, y_train = _data_converter(X_train, y_train)
    X_valid, y_valid = _data_converter(X_valid, y_valid)

    return X_train, y_train, X_valid, y_valid

def _generate_data_loaders(X_train, y_train, X_valid, y_valid, batch_size=16):

    train = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    valid = TensorDataset(X_valid, y_valid)
    valid_loader = DataLoader(valid, batch_size=len(valid.tensors[0]), shuffle=True, drop_last=True)

    return train_loader, valid_loader

def _data_converter(x, y, variable=False):

    features = torch.from_numpy(np.vstack(np.array(x)).astype(np.float32))
    if torch.isnan(features).sum() != 0:
        print("Remove NaNs from features")        
    if variable:
        features = Variable(features)

    if y is not None:
        targets = torch.from_numpy(np.vstack(np.array(y)))
        if torch.isnan(targets).sum() != 0:
            print("remove NaNs from target")
        if variable:
            targets = Variable(targets)
            
        return features, targets

    else:
        return features

def _save(obj, save_path=None, file="model.pkl"):
    """
    Saves the current instance of the class to a pickle file.
    """
    if save_path is None:
        save_path = obj.output_folder

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    with open(os.path.join(save_path, file), "wb") as f:
        dill.dump(obj, f)

    print("Saved to", os.path.join(save_path, file))

def _load(load_path=None):
    """
    Loads a saved instance of the class from a pickle file.
    """
    sys.path.append(os.path.dirname(__file__))
    with open(load_path, "rb") as f:
        return dill.load(f)

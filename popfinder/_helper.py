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
from sklearn.metrics import r2_score, root_mean_squared_error
import matplotlib.pyplot as plt

class GenPropData(Dataset):
    # Note: add option to return to old genprop data loader for 
    # data sets where sample proportions match field props?
    def __init__(self, length, data_in, row_size, zero_inflation_prob=0.1):
        self.length = length
        self.dataframe = data_in
       
        # do some handling on pop names to create a mapping (torch doesn't support
        # string, so need integer class mapping) 
        categorical_series = data_in["pop"].astype('category')
        self.pop_names = categorical_series.cat.categories.tolist()
        self.pop_num = len(self.pop_names)
        integer_labels = categorical_series.cat.codes.values
        self.unique_pops = list(range(self.pop_num))
        self.pop_to_indices = {
            pop_code: torch.tensor(np.where(integer_labels == pop_code)[0], dtype=torch.long)
            for pop_code in self.unique_pops
        }
        
        self.row_size = row_size
        self.pop_order = np.sort(pd.unique(data_in["pop"]))
        #self.unique_pops = pd.unique(data_in["pop"]).tolist()
        self.snp_length = len(data_in["alleles"].iloc[0])
        self.zero_inflation_prob = zero_inflation_prob
        #self.pop_to_indices = {
        #    pop: torch.tensor(data_in.index[data_in["pop"] == pop].tolist(), dtype=torch.long)
        #    for pop in self.unique_pops
        #}
        self.unique_pops = list(range(self.pop_num))
        self.pop_to_indices = {
            pop_code: torch.tensor(np.where(integer_labels == pop_code)[0], dtype=torch.long)
            for pop_code in self.unique_pops
        }

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # inflate zeroes by randomly dropping some pops
        # then use a random pull from a 0>x<1 distro within a pop if it is included 
        # (not just a random sample of n rows)
        # "zero-inflated direchlet"
        
        active_mask = np.random.rand(self.pop_num) > self.zero_inflation_prob
        
        # ensure not all pops are 0!
        if not np.any(active_mask):
          active_mask[np.random.randint(0, self.pop_num)] = True
        
        active_indices = np.where(active_mask)[0]
        num_active = len(active_indices)
        
        # initialize at zero
        direchlet_targets = np.zeros(self.pop_num)
        
        # set only active pops to 1, sample them
        alpha = np.ones(num_active)
        active_props = np.random.dirichlet(alpha)
        
        # add to target vector
        direchlet_targets[active_indices] = active_props
        
        # get appropriate discrete sample numbers, based on row-size
        counts = np.floor(direchlet_targets * self.row_size).astype(int)
        
        # correct any rounding issues that drop actual number of rows below row size
        remainder = self.row_size - counts.sum()
        if remainder > 0:
            # Only distribute remainders to classes that were meant to be active
            fractional_parts = (direchlet_targets * self.row_size) - counts
            # Mask out naturally inactive classes by forcing their remainder to 0
            fractional_parts[~active_mask] = -1.0
            top_indices = np.argsort(fractional_parts)[::-1][:remainder]
            counts[top_indices] += 1
        
        matrix_chunks = []
        for pop_id, count in enumerate(counts):
            if count == 0:
                continue
            available_idx = self.pop_to_indices[pop_id]
            sampled_subset = available_idx[
                torch.randint(0, len(available_idx), (count,))
            ]
            #print(np.stack(self.dataframe["alleles"].iloc[sampled_subset].values).shape)
            matrix_chunks.append(np.stack(self.dataframe["alleles"].iloc[sampled_subset].values))
        
        #x = torch.cat(matrix_chunks, dim=0)
        #print(matrix_chunks)
        #for_conv = self.dataframe.sample(n=self.row_size)
        arr = np.concatenate(matrix_chunks, axis=0)
        #print(arr.shape)
        indices = np.lexsort(arr.T[::-1])
        sorted_arr = arr[indices]
        #row_order = for_conv.index[indices]
        pop_num = self.pop_num
        #print(sorted_arr)
        # get props for each set
        #props = []
        #for i in range(pop_num):
        #  props.append(for_conv["pop"].to_list().count(self.pop_order[i]))
        #props = np.array(props)/self.row_size
        props = np.array(counts)/self.row_size

        x = torch.from_numpy(sorted_arr).float().unsqueeze(0)
        y = torch.from_numpy(props).float()
        return x, y

def plot_pop_fractions(fraction_df,popnames, title='Population fraction estimates', save_name="pop_fractions.png"):
    x = np.array(popnames)
    y = np.mean(fraction_df, axis=0).squeeze(0) # get average? Or median?
    e = np.std(fraction_df, axis=0).squeeze(0)
    plt.errorbar(x, y, e, linestyle='None', marker='^')
    plt.savefig(save_name)

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
            #nn.Softmax(dim=1) # changed to run softmax only when calculating loss
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x
        
import torch
import torch.nn as nn
import torch.nn.functional as F

class TrainableWeightedLoss(nn.Module):
    def __init__(self, initial_alpha=0.5):
        super(TrainableAlphaAdmixtureLoss, self).__init__()
        self.mae = nn.L1Loss()
        self.kldiv = nn.KLDivLoss(reduction='batchmean')
        
        # We store a raw unconstrained parameter (logit). 
        # inverse_sigmoid maps our starting alpha to its unconstrained space.
        init_logit = torch.logit(torch.tensor(initial_alpha))
        self.alpha_logit = nn.Parameter(init_logit)

    def forward(self, p_pred_logits, p_true_probs):
        # 1. Constrain alpha between 0 and 1 using Sigmoid
        alpha = torch.sigmoid(self.alpha_logit)
        
        # 2. Compute individual losses
        p_pred_probs = F.softmax(p_pred_logits, dim=-1)
        p_pred_log_probs = F.log_softmax(p_pred_logits, dim=-1)
        
        loss_mae = self.mae(p_pred_probs, p_true_proportions)
        loss_kld = self.kldiv(p_pred_log_probs, p_true_probs)
        
        # 3. Combine using the trainable alpha
        # MAetic weight = alpha, KL weight = (1 - alpha)
        total_loss = alpha * loss_mae + (1.0 - alpha) * loss_kld
        
        # Return total loss, individual metrics, and current alpha for tracking
        return total_loss, loss_mae.detach(), loss_kld.detach(), alpha.item()

class DynamicWeightedLoss(nn.Module):
    def __init__(self):
        super(DynamicWeightedLoss, self).__init__()
        self.mae = nn.L1Loss()
        self.kldiv = nn.KLDivLoss(reduction='batchmean')
        
        # Learnable parameters for log variance (s = log(sigma^2))
        self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(self, p_pred_logits, p_true_probs):
        p_pred_probs = torch.softmax(p_pred_logits, dim=-1)
        
        p_pred_log_probs = torch.log_softmax(p_pred_logits, dim=-1)
        
        loss_mae = self.mae(p_pred_probs, p_true_probs)
        loss_kld = self.kldiv(p_pred_log_probs, p_true_probs)
        
        #log_var_mae = self.log_vars[0]
        #log_var_kld = self.log_vars[1]
        
        log_var_mae = torch.clamp(self.log_vars[0], min=-2.0, max=5.0)
        log_var_kld = torch.clamp(self.log_vars[1], min=-2.0, max=5.0)
        
        precision_mae = torch.exp(-log_var_mae)
        precision_kld = torch.exp(-log_var_kld)
        
        weighted_mae = precision_mae * loss_mae + log_var_mae
        weighted_kld = precision_kld * loss_kld + log_var_kld
        
        total_loss = weighted_mae + weighted_kld
        return total_loss, loss_mae.detach(), loss_kld.detach()

class permutation_invar_nn(nn.Module):
    def __init__(self, num_cols, hidden_size, output_size):
        super().__init__()
        self.hidden_layer = nn.Linear(in_features=num_cols, out_features=hidden_size)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.sfm = nn.Softmax(dim=2)

    def forward(self, x):
        col_means = torch.mean(x, dim=2, keepdim=True)
        col_means = col_means.squeeze(1)
        
        #print(col_means.shape)
        
        out = self.hidden_layer(col_means)
        
        #print(out.shape)
        out = self.relu(out)
        out = self.output_layer(out)
        #out = self.sfm(out)
        out = out.squeeze()
        #print(out.shape)

        return out

def train_loop(dataloader, model, criterion, optimizer, batch_size):
    size = len(dataloader.dataset)
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        #print(y)
        X, y = X.to(device), y.to(device)
        pred1 = model(X)
        #pred = torch.exp(torch.log_softmax(pred1, dim=-1))
        #criterion(, y_target)
        #for_loss = torch.log_softmax(pred1, dim=-1)
        #loss = loss_fn(torch.log_softmax(pred1, dim=-1), y)
        
        #for_mae = torch.softmax(pred1, dim=-1)
        #print("True Target Matrix vs Model Predictions:")
        #print("------------------------------------------")
        #for i in range(min(3, X.size(0))):
        #    print(f"Sample {i+1} True: {y[i].cpu().numpy().round(2)}")
        #    print(f"Sample {i+1} Pred: {pred[i].cpu().detach().numpy().round(2)}\n")
        #print(f"MAE: {mae.cpu().detach().numpy().round(3)}\n")
        # Backpropagation
        total_loss, mae, kld_loss = criterion(
        p_pred_logits=pred1, 
        p_true_probs=y
        )
        total_loss.backward()
        
        #loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            print_loss, current = total_loss.item(), batch * batch_size + len(X)
            print(f"loss: {print_loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(f"MAE: {mae.cpu().detach().numpy().round(3)}\n")
            print(f"KLDIV: {kld_loss.cpu().detach().numpy().round(3)}")
    return total_loss

def test_loop(dataloader, model, criterion):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_loss, mae, kld_loss, correct = 0, 0, 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            #for_mae = torch.softmax(pred, dim=-1)
            #for_loss = torch.log_softmax(pred, dim=-1)
            #test_loss += loss_fn(torch.log_softmax(pred, dim=-1), y).item()
            pred_t = torch.exp(torch.log_softmax(pred, dim=-1))
            test_loss_it, mae_it, kld_loss_it = criterion( 
                p_pred_logits=pred, 
                p_true_probs=y
            )
            test_loss += test_loss_it
            mae += mae_it
            kld_loss+=kld_loss_it
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            correct = torch.mean(torch.nn.functional.pairwise_distance(pred_t,y, p=2))
            # also do root-mean-squared error and r2; note pred/y param order are flipped for these functions
            rsq = r2_score(y.cpu(),pred_t.cpu()) 
            rmse = root_mean_squared_error(y.cpu(),pred_t.cpu())
            #mae = torch.abs(torch.softmax(pred_t, dim=-1) - y).mean()
    test_loss /= num_batches
    correct /= size
    mae /= num_batches
    kld_loss /= num_batches
    print("End epoch test")
    print("--------------")
    print(f"test loss: {test_loss:>7f}")
    print(f"MAE: {mae.cpu().detach().numpy().round(3)}\n")
    print(f"KLDIV: {kld_loss.cpu().detach().numpy().round(3)}")
    return correct, test_loss,rsq,rmse, mae, kld_loss

def get_unknown_mixture(dataset, model, loss_fn, row_size, iterations=100):
    # add number of sampling runs as a set-able param?
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        prop_list = []
        for samp in range(iterations):
            X = torch.from_numpy(np.stack(dataset["alleles"].sample(n=row_size))).float().unsqueeze(0).unsqueeze(0)
            #print(X.shape)
            X = X.to(device)
            pred = model(X)
            pred_t = torch.exp(torch.log_softmax(pred, dim=-1))
            prop_list.append(pred_t.cpu().numpy())
    return prop_list

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

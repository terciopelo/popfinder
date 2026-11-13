import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
import itertools
import os

def _plot_training_curve(train_history, nn_type, output_folder, 
                         save, facet_by_split_rep, y_axis_zero):

    plot_data = train_history.rename(columns={"valid_loss": "validation",
                                              "train_loss": "train"})
    plot_data = pd.melt(plot_data, id_vars=["epoch", "split", "rep"], 
            value_vars=["train", "validation"], 
            var_name="dataset", value_name="loss")

    d = {'color': ['C0', 'k'], "ls" : ["-","--"]}

    if facet_by_split_rep:
        g = sns.FacetGrid(plot_data, col="split", row="rep", 
                          hue="dataset", hue_kws=d, height=2, aspect=1.5)
    else:
        g = sns.FacetGrid(plot_data, hue="dataset", hue_kws=d, height=2, aspect=1.5)

    if y_axis_zero:
        g.set(ylim=(0, max(plot_data["loss"])))
    
    g.map(sns.lineplot, "epoch", "loss", lw=0.5)
    g.add_legend(title="")

    if save:
        g.savefig(os.path.join(output_folder, nn_type + "_training_curve.png"),
            bbox_inches="tight")

def _plot_confusion_matrix(test_results, confusion_matrix, nn_type,
    output_folder, save):

    true_labels = test_results["true_pop"] # test w/ classifier

    cm = np.round(confusion_matrix, 2)
    plt.style.use("default")
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.ylabel("True Population")
    plt.xlabel("Predicted Population")
    plt.title("Confusion Matrix")
    tick_marks = np.arange(len(np.unique(true_labels)))
    plt.xticks(tick_marks, np.unique(true_labels))
    plt.yticks(tick_marks, np.unique(true_labels))
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(output_folder, nn_type + "_confusion_matrix.png"))

def _plot_assignment(e_preds, col_scheme, output_folder,
    nn_type, save, best_model_only):
        
    # e_preds.set_index("sampleID", inplace=True)

    # if nn_type == "regressor":
    #     e_preds = pd.get_dummies(e_preds["classification"], dtype=float)
    # elif nn_type == "classifier":
    #     e_preds = pd.get_dummies(e_preds["assigned_pop"], dtype=float)

    # if not best_model_only:
    #     e_preds = e_preds.reset_index().groupby("sampleID").mean()

    num_classes = len(e_preds.columns)

    sns.set()
    sns.set_style("ticks")
    e_preds.plot(kind="bar", stacked=True,
        colormap=ListedColormap(sns.color_palette(col_scheme, num_classes)),
        figsize=(12, 6), grid=None)
    legend = plt.legend(
        loc="center right",
        bbox_to_anchor=(1.2, 0.5),
        prop={"size": 15},
        title="Predicted Population",
    )
    plt.setp(legend.get_title(), fontsize="x-large")
    plt.xlabel("Sample ID", fontsize=20)
    plt.ylabel("Frequency of Assignment", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(
            output_folder, nn_type + "_assignment.png"),
            bbox_inches="tight")

def _plot_structure(preds, col_scheme, nn_type, output_folder, save):

    num_classes = len(preds.index)

    sns.set()
    sns.set_style("ticks")
    preds.plot(kind="bar", stacked=True,
        colormap=ListedColormap(sns.color_palette(col_scheme, num_classes)),
        figsize=(12, 6), grid=None)
    legend = plt.legend(loc="center right", bbox_to_anchor=(1.2, 0.5),
        prop={"size": 15}, title="Predicted Pop")
    plt.setp(legend.get_title(), fontsize="x-large")
    plt.xlabel("Actual Pop", fontsize=20)
    plt.ylabel("Frequency of Assignment", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if save:
        plt.savefig(os.path.join(output_folder,
            nn_type + "_structure.png"),
            bbox_inches="tight")


def _plot_mixture_comparison(
    estimates,
    intervals,
    holdout_mae,
    cnn_test_mae,
    classifier_cv_mae,
    output_folder,
    save=True,
    filename="mixture_method_comparison.png",
    figsize=None,
):
    """Plot aligned population-fraction estimates from multiple methods."""
    populations = list(estimates.columns)
    methods = list(estimates.index)
    if not methods or not populations:
        raise ValueError("At least one method and one population are required")

    if figsize is None:
        figsize = (max(9, 1.5 * len(populations)), 6)
    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(len(populations))
    bar_width = 0.8 / len(methods)
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * bar_width
    colors = sns.color_palette("colorblind", n_colors=len(methods))

    for method_index, method in enumerate(methods):
        values = estimates.loc[method, populations].to_numpy(dtype=float)
        bar_positions = positions + offsets[method_index]
        ax.bar(
            bar_positions,
            values,
            width=bar_width,
            label=method,
            color=colors[method_index],
        )

        if method in intervals:
            lower, upper = intervals[method]
            lower_error = np.maximum(values - lower, 0)
            upper_error = np.maximum(upper - values, 0)
            ax.errorbar(
                bar_positions,
                values,
                yerr=np.vstack([lower_error, upper_error]),
                fmt="none",
                ecolor="black",
                elinewidth=1,
                capsize=3,
            )

        if method in holdout_mae:
            mae = holdout_mae[method]
            ax.errorbar(
                bar_positions,
                values,
                yerr=np.vstack([
                    np.minimum(values, mae),
                    np.minimum(1 - values, mae),
                ]),
                fmt="none",
                ecolor=colors[method_index],
                elinewidth=1.5,
                capsize=3,
            )

        if method in cnn_test_mae:
            mae = cnn_test_mae[method]
            ax.errorbar(
                bar_positions,
                values,
                yerr=np.vstack([
                    np.minimum(values, mae),
                    np.minimum(1 - values, mae),
                ]),
                fmt="none",
                ecolor=colors[method_index],
                elinewidth=1.5,
                capsize=3,
            )

        if method in classifier_cv_mae:
            mae = classifier_cv_mae[method]
            ax.errorbar(
                bar_positions,
                values,
                yerr=np.vstack([
                    np.minimum(values, mae),
                    np.minimum(1 - values, mae),
                ]),
                fmt="none",
                ecolor=colors[method_index],
                elinewidth=1.5,
                capsize=3,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(populations, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Estimated mixture fraction")
    ax.set_xlabel("Population")
    empirical_error_sources = []
    if holdout_mae:
        empirical_error_sources.append("mixture hold-out MAE")
    if cnn_test_mae:
        empirical_error_sources.append("CNN test MAE")
    if classifier_cv_mae:
        empirical_error_sources.append("Popfinder k-fold MAE")

    if intervals and empirical_error_sources:
        ax.set_title(
            "Population mixture estimates by method\n"
            "Black: Bayesian 95% credible intervals; colored: "
            + " and ".join(empirical_error_sources)
            + " (MAE bars are not confidence intervals)"
        )
    elif intervals:
        ax.set_title(
            "Population mixture estimates by method\n"
            "Error bars show Bayesian 95% credible intervals"
        )
    elif empirical_error_sources:
        ax.set_title(
            "Population mixture estimates by method\n"
            "Colored error bars show "
            + " and ".join(empirical_error_sources)
            + " (not confidence intervals)"
        )
    else:
        ax.set_title("Population mixture estimates by method")
    ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    if save:
        os.makedirs(output_folder, exist_ok=True)
        fig.savefig(os.path.join(output_folder, filename), bbox_inches="tight")

    return fig

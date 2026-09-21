import plotly.graph_objects as go
import plotly.io as pio

from .general_utils import create_new_filename, smape
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torch
from pathlib import Path


def torch_to_series(x: torch.Tensor, dates, name=None):
    assert len(x) == len(dates), f"x and dates must have the same length, got {len(x)} and {len(dates)}"
    x = x.detach().cpu().flatten().numpy()
    dates = pd.DatetimeIndex(pd.to_datetime(dates, errors="raise"))

    return pd.Series(x, index=dates, name=name)


def add_config_to_plot(fig, run_config: dict):
    """ Add a comment at the bottom of the plot with config params """
    fig.tight_layout(rect=[0, 0.15, 1, 0.95])  # [left, bottom, right, top]
    footer_text = " | ".join(f"{k}: {v}" for k, v in run_config.items())
    fig.text(
        0.02, 0.02, f"Training config: {footer_text}",
        fontsize=8,
        fontfamily='monospace',
        ha='left',
        wrap=True,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )


class TrainingTracker:
    def __init__(self, config: dict, tracking_dir, fold_span):
        self.epochs = []
        self.train_losses = []
        self.loss_contribs = []
        self.val_losses = []
        self.lrs = []
        self.train_config = config
        self.fold_span = fold_span

        # Saving preparation
        tracking_dir.mkdir(parents=True, exist_ok=True)
        self.tracking_dir = tracking_dir

    def add_epoch(self, epoch, train_loss, loss_contribs, val_loss, lr):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.loss_contribs.append(loss_contribs)
        self.val_losses.append(val_loss)
        self.lrs.append(lr)


    # def to_dataframe(self):
    #     return pl.DataFrame({
    #         'epoch': self.epochs,
    #         'train_loss': self.train_losses,
    #         'val_loss': self.val_losses,
    #         'lr': self.lrs
    #     })


    def aggregate_contribs(self):
        # assume self.loss_contribs = list of epoch dicts
        all_keys = self.loss_contribs[0].keys()

        aggregated = {k: [] for k in all_keys}

        for epoch_dict in self.loss_contribs:
            for k in all_keys:
                aggregated[k].append(np.mean(epoch_dict[k]))

        return aggregated

    
    def plot_contribs(self, show=False):
        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss Contributions")

        contribs = self.aggregate_contribs()

        labels = list(contribs.keys())
        values = [contribs[k] for k in labels]

        ax1.stackplot(
            self.epochs,
            values,
            labels=labels,
            alpha=0.8,
        )

        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right")

        contribs_path = create_new_filename(
            self.tracking_dir,
            "train_loss_contribs",
            "png"
        )

        fig.savefig(contribs_path, dpi=100, bbox_inches="tight")

        if show:
            plt.show()

        plt.close(fig)


    def plot_losses(self, show=False):
            """
            Simple plot of train vs validation loss through epochs
            """

            loss_type = self.train_config["loss_type"]

            # Plot curves
            # fig, ax1 = plt.subplots(figsize=(12, 6))
            fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

            # Plot train loss on left y-axis
            color = 'tab:blue'
            # ax1.set_xlabel('Epoch')
            ax1.set_ylabel(f'Train {loss_type}', color=color)
            ax1.plot(self.epochs, self.train_losses, color=color, linewidth=2, label=f'Train {loss_type}')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, alpha=0.3)
            
            # Create second y-axis for validation loss
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel(f'Validation {loss_type}', color=color)
            ax2.plot(self.epochs, self.val_losses, color=color, linewidth=2, 
                    linestyle='--', label=f'Validation {loss_type}')
            ax2.tick_params(axis='y', labelcolor=color)
            
            # Apply log scale to both axes if needed
            for ax, losses in [(ax1, self.train_losses), (ax2, self.val_losses)]:
                if len(losses) > 1 and all(l > 0 for l in losses):  # All positive
                    ratio = max(losses) / min(losses)
                    if ratio > 100:
                        ax.set_yscale('log')           

            n_epochs = len(self.epochs)
            max_ticks = 10  # maximum number of ticks you want on x-axis
            step = max(1, int(np.ceil(n_epochs / max_ticks))) # automatically compute a “nice” step
            ax1.set_xticks(np.arange(0, n_epochs + 1, step))
            # ax1.set_xticks(np.arange(0, len(self.epochs) + 1, 5))

            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.plot(self.epochs, self.lrs, color='tab:green', linewidth=2, label='LR')
            ax3.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax3.grid(True, alpha=0.3)

            fig.suptitle(f"Evolution of Training and Validation Losses for {[d.strftime('%Y-%m-%d') for d in self.fold_span]}")

            # Combine legends from both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            add_config_to_plot(fig, self.train_config)
            
            # Save the plot
            train_conv_path = create_new_filename(self.tracking_dir, "train_conv", "png")
            fig.savefig(train_conv_path, dpi=100, bbox_inches='tight')
            
            plt.close(fig)


def plot_grid_search_results(x_label, y_label, data: pd.DataFrame, config: dict, tracking_dir: Path):
    loss_type = config["loss_type"]
    
    vars = data.columns
    pivot_table = data.pivot(columns=vars[0], index=vars[1], values=vars[2])

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use imshow for heatmap
    im = ax.imshow(
        pivot_table.values,
        aspect='auto',
        cmap='jet',
    )

    # Add labels and title
    fig.suptitle('Results of Grid Scan', fontsize=14, pad=20)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    x_ticks = np.arange(len(pivot_table.index))
    y_ticks = np.arange(len(pivot_table.columns))

    ax.set_xticks(y_ticks)
    ax.set_yticks(x_ticks)
    ax.set_xticklabels(pivot_table.columns)
    ax.set_yticklabels(pivot_table.index)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f'{loss_type} Loss', fontsize=12)

    # Add text annotations for each cell
    mean_val = np.nanmean(pivot_table.values)
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            v = pivot_table.iloc[i, j]
            if not np.isnan(v):
                ax.text(
                    j, i, f'{v:.2e}',
                    ha='center', va='center',
                    color='white' if v < mean_val else 'black',
                    fontsize=9
                )

    # Add config to plot
    add_config_to_plot(fig, config)

    # Save the plot
    save_path = create_new_filename(tracking_dir, "grid_search_results", "png")
    fig.savefig(save_path)
    plt.close(fig)
    

def plot_val_predictions_direction(
        y_test,
        y_pred,
        tracking_dir,
        probas=None,
        title="Prediction VS Real",
        n_plot=None,
    ):
    """
    Directional prediction plots.

    Expects y_test and y_pred to be pandas Series with dates as index.
    Extra dates in y_pred are plotted in the time plot.
    Confusion matrix is computed only where y_test and y_pred overlap.
    """
    label_names = ["Short", "Neutral", "Long"]
    label_to_idx = {-1: 0, 0: 1, 1: 2}

    # -----------------------------
    # Confusion matrix on overlap only
    # -----------------------------
    y_test_common, y_pred_common = y_test.align(y_pred, join="inner")

    cm_df = pd.concat(
        [y_test_common.rename("y_test"), y_pred_common.rename("y_pred")],
        axis=1,
    ).dropna()

    y_test_i = cm_df["y_test"].astype(int).to_numpy()
    y_pred_i = cm_df["y_pred"].astype(int).to_numpy()

    cm = np.zeros((3, 3), dtype=int)

    for true_val, pred_val in zip(y_test_i, y_pred_i):
        if true_val in label_to_idx and pred_val in label_to_idx:
            cm[label_to_idx[true_val], label_to_idx[pred_val]] += 1

    fig, ax = plt.subplots(figsize=(8, 8))
    cm_plot = cm[::-1, :]
    im = ax.imshow(cm_plot, cmap="jet")

    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(label_names)
    ax.set_yticklabels(label_names[::-1])

    ax.set_xlabel("Predicted Direction")
    ax.set_ylabel("True Direction")
    ax.set_title("NN directional predictions")

    max_cm = cm_plot.max()
    for row in range(3):
        for col in range(3):
            val = cm_plot[row, col]
            ax.text(
                col,
                row,
                str(val),
                ha="center",
                va="center",
                color="white" if max_cm == 0 or val < max_cm * 0.25 else "black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    save_path = create_new_filename(tracking_dir, "confusion_pred_vs_true", "png")
    fig.savefig(save_path)
    plt.close(fig)

    # -----------------------------
    # True target vs predict by date
    # -----------------------------
    dates = y_pred.index
    y_test_plot = y_test.reindex(dates)
    y_pred_plot = y_pred

    if n_plot is None:
        n_plot = len(dates)
    else:
        n_plot = min(n_plot, len(dates))

    dates_plot = dates[:n_plot]
    y_test_plot = y_test_plot.iloc[:n_plot]
    y_pred_plot = y_pred_plot.iloc[:n_plot]
    
    # Matplotlib can handle np.nan, but it often chokes on pandas NAType.
    y_test_plot = pd.to_numeric(y_test_plot, errors="coerce").astype(float) 
    y_pred_plot = pd.to_numeric(y_pred_plot, errors="coerce").astype(float)

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        dates_plot,
        y_test_plot + 0.02,
        linestyle="None",
        marker="o",
        markersize=3,
        label="True",
        color="tab:blue",
        alpha=0.75,
    )

    scatter_kwargs = dict(
        marker="x",
        label="Predicted",
        color="tab:red",
        alpha=0.75,
    )

    if probas is not None:
        if isinstance(probas, pd.DataFrame):
            probas_plot = probas.reindex(dates_plot).to_numpy()
        else:
            probas_plot = np.asarray(probas)[:n_plot]

        pred_class = (y_pred_plot.astype(int).to_numpy() + 1).clip(0, 2)
        pred_proba = probas_plot[np.arange(n_plot), pred_class]
        scatter_kwargs["s"] = 12 + 45 * pred_proba
    else:
        scatter_kwargs["s"] = 25

    ax.scatter(
        dates_plot,
        y_pred_plot - 0.02,
        **scatter_kwargs,
    )

    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Short", "Neutral", "Long"])
    ax.set_ylim(-1.35, 1.35)
    ax.axhline(y=0, color="k", linestyle="--", linewidth=1, alpha=0.45)

    ax.set_xlabel("Date")
    ax.set_ylabel("Target")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    save_path = create_new_filename(tracking_dir, "pred_vs_true_by_sample", "png")
    fig.savefig(save_path)
    plt.close(fig)


# def plot_directional_error(dates, y_test, y_pred, n_plot, tracking_dir):
#     y_test = np.asarray(y_test[:n_plot]).astype(int)
#     y_pred = np.asarray(y_pred[:n_plot]).astype(int)
#     idx = dates[:n_plot]

#     errors = y_test - y_pred
#     accuracy = np.mean(errors == 0)
#     severe_miss = np.mean(np.abs(errors) == 2)

#     fig, axs = plt.subplots(3, 1, figsize=(12, 7))

#     # --- Subplot 1: class error over time ---
#     axs[0].scatter(
#         idx,
#         errors,
#         s=8,
#         alpha=0.75,
#         color="tab:red",
#     )
#     axs[0].axhline(0, color="k", linestyle="--", linewidth=1)
#     axs[0].set_yticks([-2, -1, 0, 1, 2])
#     axs[0].set_ylabel("True - Pred")
#     axs[0].set_title("Directional Classification Error Over Time")
#     axs[0].text(
#         0.01,
#         1.02,
#         f"accuracy={accuracy:.1%}, severe miss={severe_miss:.1%}",
#         transform=axs[0].transAxes,
#         fontsize=10,
#         verticalalignment="bottom",
#     )

#     # --- Subplot 2: distribution of class errors ---
#     error_bins = np.arange(-2.5, 3.0, 1.0)
#     axs[1].hist(
#         errors,
#         bins=error_bins,
#         color="tab:cyan",
#         alpha=0.7,
#         edgecolor="k",
#     )
#     axs[1].axvline(0, color="k", linestyle="--", linewidth=1)
#     axs[1].set_xticks([-2, -1, 0, 1, 2])
#     axs[1].set_xlabel("Class Error")
#     axs[1].set_ylabel("Frequency")
#     axs[1].set_title("Distribution of Directional Errors")

#     # --- Subplot 3: true vs predicted class distribution ---
#     labels = np.array([-1, 0, 1])
#     x = np.arange(len(labels))
#     width = 0.38

#     true_counts = np.array([(y_test == label).sum() for label in labels])
#     pred_counts = np.array([(y_pred == label).sum() for label in labels])

#     axs[2].bar(
#         x - width / 2,
#         true_counts,
#         width,
#         label="True",
#         color="tab:blue",
#         alpha=0.75,
#     )
#     axs[2].bar(
#         x + width / 2,
#         pred_counts,
#         width,
#         label="Predicted",
#         color="tab:red",
#         alpha=0.75,
#     )
#     axs[2].set_xticks(x)
#     axs[2].set_xticklabels(["Short", "Neutral", "Long"])
#     axs[2].set_xlabel("Class")
#     axs[2].set_ylabel("Frequency")
#     axs[2].set_title("True vs Predicted Class Distribution")
#     axs[2].legend()

#     fig.autofmt_xdate()
#     fig.tight_layout()

#     save_path = create_new_filename(tracking_dir, "directional_prediction_error", "png")
#     fig.savefig(save_path, dpi=300)
#     plt.close(fig)


def plot_val_predictions(
        y_test, 
        y_pred, 
        tracking_dir, 
        n_plot=None,
        # n_plot:int = 400 #200
    ):
    """
    Scatter plot + time series plot of predictions vs true values.

    Expects y_test and y_pred to be pandas Series with dates as index.
    """

    # True target vs predict: first [n_plot] points (for clarity)
    plot_pred_vs_target(y_test, y_pred, n_plot, tracking_dir)

    # Scatter
    scatter_pred_real(y_test, y_pred, n_plot, tracking_dir)

    # Relative error
    plot_rel_error(y_test, y_pred, n_plot, tracking_dir)


def plot_pred_vs_target(y_test, y_pred, n_plot, tracking_dir):
    """
    True target vs predict.

    y_test and y_pred should be pandas Series with dates as index.
    Extra dates in y_pred are still plotted.
    Metrics are computed only where y_test and y_pred overlap.
    """
    dates = y_pred.index

    y_test_plot = y_test.reindex(dates)
    y_pred_plot = y_pred

    if n_plot is None:
        n_plot = len(dates)
    else:
        n_plot = min(n_plot, len(dates))

    dates_plot = dates[:n_plot]
    y_test_plot = y_test_plot.iloc[:n_plot]
    y_pred_plot = y_pred_plot.iloc[:n_plot]

    common_mask = y_test_plot.notna() & y_pred_plot.notna()

    y_test_common = y_test_plot.loc[common_mask]
    y_pred_common = y_pred_plot.loc[common_mask]

    y_abs_avg = np.mean(np.abs(y_test_common))

    if True:

        fig, ax = plt.subplots(figsize=(14, 5))

        ax.plot(
            dates_plot,
            y_test_plot,
            'b-',
            label='True',
            alpha=0.7,
            linewidth=1,
        )
        ax.plot(
            dates_plot,
            y_pred_plot,
            'r-',
            label='Predicted',
            alpha=0.7,
            linewidth=1,
        )
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.7)

        mse = np.mean((y_pred_common - y_test_common) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_pred_common - y_test_common))
        avg_perc_error = mae / np.mean(np.abs(y_test_common)) * 100

        ax.text(
            0.02, 0.98,
            f'RMSE: {rmse:.2e} ({rmse / y_abs_avg * 100:.2f}%)\nrel MAE: {avg_perc_error:.2f}%',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        ax.set_xlabel('Date')
        ax.set_ylabel('Target')
        ax.set_title("Prediction VS Real")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()

        save_path = create_new_filename(tracking_dir, "pred_vs_true_by_sample", "png")
        fig.savefig(save_path)

        plt.close(fig)
    
    else:
        fig = go.Figure()

        # True
        fig.add_trace(
            go.Scatter(
                x=dates_plot,
                y=y_test_plot,
                mode="lines",
                name="True",
                line=dict(
                    color="blue",
                    width=1,
                ),
                opacity=0.7,
            )
        )

        # Predicted
        fig.add_trace(
            go.Scatter(
                x=dates_plot,
                y=y_pred_plot,
                mode="lines",
                name="Predicted",
                line=dict(
                    color="red",
                    width=1,
                ),
                opacity=0.7,
            )
        )

        # Zero line
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_width=1,
            opacity=0.7,
        )


        # Metrics
        mse = np.mean((y_pred_common - y_test_common) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_pred_common - y_test_common))
        avg_perc_error = mae / np.mean(np.abs(y_test_common)) * 100


        # Add metrics box
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=(
                f"RMSE: {rmse:.2e} "
                f"({rmse / y_abs_avg * 100:.2f}%)<br>"
                f"rel MAE: {avg_perc_error:.2f}%"
            ),
            showarrow=False,
            align="left",
            bgcolor="white",
            opacity=0.8,
        )


        fig.update_layout(
            title="Prediction VS Real",
            xaxis_title="Date",
            yaxis_title="Target",
            template="plotly_white",
            height=500,
            hovermode="x unified",
            legend=dict(
                x=0,
                y=1,
            ),
        )


        save_path = create_new_filename(
            tracking_dir,
            "pred_vs_true_by_sample",
            "json",
        )

        pio.write_json(fig, save_path)


def scatter_pred_real(y_test, y_pred, n_plot, tracking_dir):
    y_test, y_pred = y_test.align(y_pred, join="inner")

    scatter_df = pd.concat(
        [y_test.rename("y_test"), y_pred.rename("y_pred")],
        axis=1,
    ).dropna()

    if n_plot is not None:
        scatter_df = scatter_df.iloc[:n_plot]

    y_test_values = scatter_df["y_test"].to_numpy()
    y_pred_values = scatter_df["y_pred"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test_values, y_pred_values, s=5, alpha=0.4)

    lo = min(y_test_values.min(), y_pred_values.min())
    hi = max(y_test_values.max(), y_pred_values.max())

    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set_xlabel("True Target")
    ax.set_ylabel("Predicted Target")
    ax.set_title("NN predictions")
    ax.grid(True, alpha=0.3)
    # add_config_to_plot(fig, config)

    save_path = create_new_filename(tracking_dir, "scatter_pred_vs_true", "png")
    fig.savefig(save_path)
    plt.close(fig)


def plot_rel_error(y_test, y_pred, n_plot, tracking_dir):
    error_df = pd.concat(
        [y_test.rename("y_test"), y_pred.rename("y_pred")],
        axis=1,
    ).dropna()

    if n_plot is not None:
        error_df = error_df.iloc[:n_plot]

    y_test = error_df["y_test"]
    y_pred = error_df["y_pred"]
    dates = error_df.index

    y_abs_avg = np.mean(np.abs(y_test))

    # Compute residuals
    errors = y_test - y_pred
    rel_errors = errors/y_abs_avg*100.
    # rel_errors = np.minimum(np.abs(rel_errors), 100.) * np.sign(rel_errors)
    # smapes = smape(y_test[:n_plot], y_pred[:n_plot])
    
    # idx = range(len(errors))
    fig, axs = plt.subplots(3, 1, figsize=(12, 6))

    # --- Subplot 1: residuals over time ---
    # plt.plot(idx, rel_errors, 'm-', alpha=0.7, linewidth=1, label='Error (True - Pred)/True')
    # Plot with dynamic mean in legend
    axs[0].plot(
        dates,
        errors,
        'r-', 
        alpha=0.7, 
        linewidth=1, 
    )    
    axs[0].set_ylabel("Error", color='r')
    axs[0].tick_params(axis='y', labelcolor='r')

    ax2 = axs[0].twinx()
    ax2.set_ylabel("Relative Error (%)", color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.set_ylim([e/y_abs_avg*100 for e in axs[0].get_ylim()])  # scale to relative error

    # plt.plot(idx, smapes, 'm-', alpha=0.7, linewidth=1, label='sMAPE')
    axs[0].axhline(0, color='k', linestyle='--', linewidth=1)
    # plt.ylabel("Error (%)")
    axs[0].set_title("Error Over Time")
    # plt.title("symmetric Mean Absolute Percentage Error")
    # plt.yscale('symlog', linthresh=0.1)  # linear between -0.1 and 0.1, log beyond
    # plt.yscale('log'); plt.ylim(bottom=1., top=250.)      # start y-axis at 1
    # axs[0].legend()
    axs[0].text(
        0.01, 1.02,  # position in axes coordinates
        f'error = (yr-yp)/avg(|yr|), mean(|error|)= {np.mean(np.abs(rel_errors)):.1f}%',
        transform=axs[0].transAxes,  # use axs[0] instead of plt.gca()
        fontsize=10,
        verticalalignment='bottom'
    )

    # --- Subplot 2: histogram of residuals ---
    xmin = min(np.min(errors), np.min(y_test))
    xmax = max(np.max(errors), np.max(y_test))
    bins = np.linspace(xmin, xmax, 50+1)
    # max_val_hist = np.ceil(max(abs(rel_errors)))
    # bin_width = 2
    # bins = np.arange(-max_val_hist - bin_width, max_val_hist + bin_width, bin_width)
    hist_counts_1, _, _ =    axs[1].hist(errors, bins=bins, color='c', alpha=0.7, edgecolor='k')
    # plt.hist(smapes, bins=30, color='c', alpha=0.7, edgecolor='k')
    axs[1].axvline(0, color='k', linestyle='--', linewidth=1)
    axs[1].axvline(np.mean(errors), color='r', linestyle='--', linewidth=1)
    axs[1].set_xlabel("Error")
    axs[1].set_ylabel("Frequency")
    axs[1].set_title("Distribution of Errors")
    # plt.title("Distribution of sMAPE")
    # plt.xticks(bins)

    # --- Subplot 3: histogram of target (for comparison) ---
    hist_counts_2, _, _ = axs[2].hist(y_test, bins=bins, color='c', alpha=0.7, edgecolor='k')
    # plt.hist(smapes, bins=30, color='c', alpha=0.7, edgecolor='k')
    axs[2].axvline(0, color='k', linestyle='--', linewidth=1)
    axs[2].axvline(np.mean(y_test), color='r', linestyle='--', linewidth=1)
    axs[2].set_xlabel("Target Distribution")
    axs[2].set_ylabel("Frequency")
    axs[2].set_title("Distribution of Target (for Comparison)")
    # plt.title("Distribution of sMAPE")
    # plt.xticks(bins)

    # Align axes of last 2 plots
    axs[1].set_xlim([xmin, xmax])
    axs[2].set_xlim([xmin, xmax])
    ymax = max(hist_counts_1.max(), hist_counts_2.max())
    axs[1].set_ylim(0, ymax)
    axs[2].set_ylim(0, ymax)

    fig.tight_layout()

    # Save to file
    save_path = create_new_filename(tracking_dir, "prediction_error", "png")
    fig.savefig(save_path, dpi=300)    
    plt.close(fig)
import torch as t
from torch import Tensor
from jaxtyping import Float
import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import Any, Iterator

from dictionary_learning.dictionary_learning import CrossCoder
from dictionary_learning.dictionary_learning.dictionary import BatchTopKCrossCoder
from dictionary_learning.dictionary_learning.training import trainSAE


@dataclass
class CrosscoderMetrics:
    save_dir: str
    device: t.device
    train_crosscoder_args: dict
    config: dict
    crosscoder: BatchTopKCrossCoder
    delta_norms: Float[Tensor, "n_features"]
    beta_reconstruction_model_0: Float[Tensor, "n_features"]
    beta_reconstruction_model_1: Float[Tensor, "n_features"]
    beta_error_model_0: Float[Tensor, "n_features"]
    beta_error_model_1: Float[Tensor, "n_features"]
    nu_reconstruction: Float[Tensor, "n_features"]
    nu_error: Float[Tensor, "n_features"]
    top_n_activations: Float[Tensor, "n_features top_n"]

    def __init__(self, save_dir: str, device: t.device) -> None:
        self.save_dir = save_dir
        self.device = device
        self.train_crosscoder_args = self.load_train_crosscoder_args(save_dir)
        self.config = self.load_config(self.save_dir)
        self.crosscoder = self.load_model(self.save_dir, device)
        self.delta_norms = self.compute_delta_norms()
        self.beta_reconstruction_model_0, self.beta_error_model_0 = self.compute_beta(
            model_i=0, device=self.device
        )
        self.beta_reconstruction_model_1, self.beta_error_model_1 = self.compute_beta(
            model_i=1, device=self.device
        )
        self.nu_reconstruction = (
            self.beta_reconstruction_model_0 / self.beta_reconstruction_model_1
        )
        self.nu_error = self.beta_error_model_0 / self.beta_error_model_1
        self.top_n_activations = self.compute_top_n_activations(top_n=9)
        self.plot_delta_norms()
        self.plot_betas()
        self.plot_nu()

    # Save + Load
    def save(self, save_dir: str) -> None:
        path = os.path.join(save_dir, "crosscoder_metric.pt")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def _load_helper(path: str):
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data
        else:
            raise FileNotFoundError(f"File {path} not found.")

    @staticmethod
    def load(save_dir: str):  # -> CrosscoderMetrics
        path = os.path.join(save_dir, "crosscoder_metric.pt")
        return CrosscoderMetrics._load_helper(path)

    @staticmethod
    def load_train_crosscoder_args(save_dir: str) -> dict:
        path = os.path.join(save_dir, "train_crosscoder_args.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"File {path} not found.")

    @staticmethod
    def load_config(save_dir: str) -> dict:
        path = os.path.join(save_dir, "config.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            return data.get("trainer", {})
        else:
            raise FileNotFoundError(f"File {path} not found.")

    @staticmethod
    def load_model(save_dir: str, device) -> CrossCoder:
        path = os.path.join(save_dir, "model_final.pt")
        if os.path.exists(path):
            return BatchTopKCrossCoder.from_pretrained(path, device=device)
        else:
            raise FileNotFoundError(f"File {path} not found.")

    def compute_delta_norms(
        self, epsilon: float = 0.0001
    ) -> Float[Tensor, "n_activations"]:
        d_model_0_vectors = self.crosscoder.decoder.weight[0]
        d_model_1_vectors = self.crosscoder.decoder.weight[1]
        norm_sq_model1 = t.sum(d_model_0_vectors**2, dim=-1)
        norm_sq_model2 = t.sum(d_model_1_vectors**2, dim=-1)
        max_norms_sq = t.maximum(norm_sq_model1, norm_sq_model2)

        delta_norms = 0.5 * (
            (norm_sq_model2 - norm_sq_model1) / (max_norms_sq + epsilon) + 1.0
        )
        return delta_norms.cpu()

    def _activations_both_generator(self, tqdm_desc) -> Iterator[t.Tensor]:
        val_dataset = PairedActivationCache(
            self.train_crosscoder_args["data_path"]["val_activations_stor_dir_model_0"],
            self.train_crosscoder_args["data_path"]["val_activations_stor_dir_model_1"],
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1000,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )
        for activations_both in tqdm(val_dataloader, desc=tqdm_desc):
            # Move activations_both to device
            activations_model_0_dev = activations_both[0].to(self.device)
            activations_model_1_dev = activations_both[1].to(self.device)
            activations_both = t.stack(
                [activations_model_0_dev, activations_model_1_dev], dim=1
            )
            yield activations_both

    def compute_beta(
        self, model_i: int, device: t.device
    ) -> tuple[Float[Tensor, "n_features"], Float[Tensor, "n_features"]]:
        """
        beta = (d.T @ (Y.T @ F)) / (||d||^2 * ||f||^2)
        """
        D = self.crosscoder.decoder.weight[model_i]
        D_chat_norms_sq = D.square().sum(dim=1)  # ||D_j||_2, [dictionary_size]

        # Accumulators for sums over all samples, for each latent j.
        # numer_r[j] = sum_samples_i ( F[i,j] * <d_j, Y_reconstruction_target[i,:]> )
        # numer_e[j] = sum_samples_i ( F[i,j] * <d_j, Y_error_target[i,:]> )
        # sum_F_sq[j] = sum_samples_i ( F[i,j]^2 ) (this is ||f_j||^2 from the formula)
        dict_size = D.shape[0]
        numer_r = t.zeros(dict_size, device=device, dtype=t.float32)  # [dict_size]
        numer_e = t.zeros_like(numer_r)  # [dict_size]
        sum_F_sq = t.zeros_like(numer_r)  # [dict_size]

        self.crosscoder.eval()
        with t.no_grad():
            for activations_both in self._activations_both_generator(
                f"Computing beta of model {model_i + 1}"
            ):
                reconstructions_both, F_batch = self.crosscoder.forward(
                    activations_both, output_features=True
                )
                activations = activations_both[
                    :, model_i, :
                ]  # [batch_size, activation_dim]
                reconstructions = reconstructions_both[
                    :, model_i, :
                ]  # [batch_size, activation_dim]

                # F_batch: sparse feature activations from sae.encode.
                sum_F_sq += F_batch.square().sum(
                    dim=0
                )  # [dict_size] (F_batch is [batch_size, dict_size])

                # Determine target Y for reconstruction (Y_r) and error (Y_e)
                # Y_r, Y_e have shape (batch_size, activation_dim)
                Y_r = reconstructions
                Y_e = activations - Y_r

                # Accumulate numerators: sum_i F[i,j] * <d_j, Y_target[i,:]>
                # (Y_target @ D.T) gives <Y_target[i,:], d_j> for each sample i and latent j  # [batch_size, dict_size]
                numer_r += (F_batch * (Y_r @ D.T)).sum(dim=0)  # [dict_size]
                numer_e += (F_batch * (Y_e @ D.T)).sum(dim=0)  # [dict_size]

        # Denominator for beta_j = ||f_j||^2 * ||d_j||^2)
        denominators = sum_F_sq * D_chat_norms_sq  # [dict_size]

        beta_r = t.nan_to_num(numer_r / denominators, nan=1.0, posinf=1.0, neginf=1.0)
        beta_e = t.nan_to_num(numer_e / denominators, nan=1.0, posinf=1.0, neginf=1.0)

        return beta_r, beta_e

    def compute_top_n_activations(top_n: int) -> Float[Tensor, "n_features top_n"]:
        """
        Creates tensor that maps from feature j to the indicies of the top n activations of that feature.
        The indices are for # TODO
        """

    def get_delta_thresholds(
        self, threshold_only: float = 0.1, threshold_shared: float = 0.1
    ):
        """
        Returns the thresholds for delta norms based on the provided parameters.
        """
        threshold_model_0_only = threshold_only
        threshold_model_1_only = 1 - threshold_only
        threshold_shared_lower = 0.5 - threshold_shared
        threshold_shared_upper = 0.5 + threshold_shared
        assert (
            threshold_model_0_only
            < threshold_shared_lower
            < threshold_shared_upper
            < threshold_model_1_only
        ), (
            f"Invalid thresholds: {threshold_model_0_only}, {threshold_shared_lower}, {threshold_shared_upper}, {threshold_model_1_only}"
        )
        return (
            threshold_model_0_only,
            threshold_model_1_only,
            threshold_shared_lower,
            threshold_shared_upper,
        )

    def plot_delta_norms(self) -> tuple[plt.Figure, list[plt.Axes]]:
        """
        Plots a histogram of delta norms with two subplots: one with linear y-axis and one with logarithmic y-axis.
        Linear same as in https://arxiv.org/pdf/2504.02922
        """
        (
            threshold_model_0_only,
            threshold_model_1_only,
            threshold_shared_lower,
            threshold_shared_upper,
        ) = self.get_delta_thresholds()

        # Data
        data_to_plot = self.delta_norms.detach().numpy()
        bins = np.linspace(0, 1, 101)  # 100 bins from 0 to 1, [0, 0.01, 0.02, ..., 1.0]

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(
            f"Histogram of Delta Norms (Features: {len(data_to_plot)})", fontsize=16
        )
        titles = ["Linear Y-Axis", "Logarithmic Y-Axis"]
        y_scales = ["linear", "log"]
        for i, ax in enumerate(axes):
            counts, _, patches = ax.hist(
                data_to_plot, bins=bins, edgecolor="black", alpha=0.7
            )
            bin_centers = (bins[:-1] + bins[1:]) / 2
            for count, patch, bin_center in zip(counts, patches, bin_centers):
                if bin_center <= threshold_model_0_only:
                    patch.set_facecolor("green")
                elif bin_center >= threshold_model_1_only:
                    patch.set_facecolor("lightblue")
                elif threshold_shared_lower <= bin_center <= threshold_shared_upper:
                    patch.set_facecolor("orange")
                else:
                    patch.set_facecolor("grey")

            ax.set_title(titles[i])
            ax.set_xlabel("Delta Norm Value")
            ax.set_ylabel("Number of Features")
            ax.set_yscale(y_scales[i])
            if y_scales[i] == "log":
                ax.set_ylim(bottom=0.1)  # Avoid log(0)
            ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save plot
        filename = f"delta_norms.png"
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, filename)
            try:
                fig.savefig(save_path)
                print(f"Delta norms plot saved to {save_path}")
            except Exception as e:
                print(f"Error saving plot: {e}")

        return fig, axes

    def plot_betas(self) -> tuple[plt.Figure, list[plt.Axes]]:
        """Plots a 2x2 grid of histograms for beta values."""
        beta_data = {
            r"$\beta^r$ Model 0 (Reconstruction)": self.beta_reconstruction_model_0.cpu()
            .detach()
            .numpy(),
            r"$\beta^\epsilon$ Model 0 (Error)": self.beta_error_model_0.cpu()
            .detach()
            .numpy(),
            r"$\beta^r$ Model 1 (Reconstruction)": self.beta_reconstruction_model_1.cpu()
            .detach()
            .numpy(),
            r"$\beta^\epsilon$ Model 1 (Error)": self.beta_error_model_1.cpu()
            .detach()
            .numpy(),
        }

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Histograms of Beta Values", fontsize=16)
        axes = axes.flatten()

        for i, (title, data) in enumerate(beta_data.items()):
            ax = axes[i]
            if len(data) > 0:
                ax.hist(data, bins=50, edgecolor="black", alpha=0.7)
                ax.set_title(title)
                ax.set_xlabel("Beta Value")
                ax.set_ylabel("Number of Features")
                ax.grid(True, linestyle="--", alpha=0.6)
                # ax.set_yscale('log')
                ax.set_ylim(bottom=0.1)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_title(title)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if self.save_dir:
            filename = "betas_histogram.png"
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, filename)
            try:
                fig.savefig(save_path)
                print(f"Betas plot saved to {save_path}")
            except Exception as e:
                print(f"Error saving betas plot: {e}")
        # plt.show() # Uncomment to display
        return fig, axes

    def plot_nu(self):
        """
        Plots nu_error vs nu_reconstruction, colored by delta_norm categories,
        Same as in https://arxiv.org/pdf/2504.02922
        """
        # Data
        nu_e_data = self.nu_error.cpu().detach().numpy()
        nu_r_data = self.nu_reconstruction.cpu().detach().numpy()
        delta_norms_data = self.delta_norms.cpu().detach().numpy()

        # Categories for each feature $j$
        (
            threshold_model_0_only,
            threshold_model_1_only,
            threshold_shared_lower,
            threshold_shared_upper,
        ) = self.get_delta_thresholds()
        categories = []
        for dn in delta_norms_data:
            if dn <= threshold_model_0_only:
                categories.append("Model 0 Only")
            elif dn >= threshold_model_1_only:
                categories.append("Model 1 Only")
            elif threshold_shared_lower <= dn <= threshold_shared_upper:
                categories.append("Shared")
            else:
                categories.append("Other")

        df = pd.DataFrame(
            {
                r"$\nu^\epsilon$ (Error)": nu_e_data,
                r"$\nu^r$ (Reconstruction)": nu_r_data,
                "Category": categories,
            }
        )

        # Plot
        palette = {
            "Model 0 Only": "cornflowerblue",
            "Shared": "grey",
            "Model 1 Only": "darkorange",
            "Other": "lightgrey",
        }
        categories_to_display = ["Model 0 Only", "Model 1 Only"]
        df_plot = df[
            df["Category"].isin(categories_to_display)
        ].copy()  # Use .copy() to avoid SettingWithCopyWarning
        if df_plot.empty:
            print("No data to plot for the selected categories in plot_nu.")
            fig, ax = plt.subplots()
            ax.text(
                0.5, 0.5, "No data for selected categories", ha="center", va="center"
            )
            if self.save_dir:
                plt.savefig(os.path.join(self.save_dir, "nu_plot_nodata.png"))
            return fig  # Return an empty figure or handle appropriately

        g = sns.JointGrid(
            data=df_plot,
            x=r"$\nu^\epsilon$ (Error)",
            y=r"$\nu^r$ (Reconstruction)",
            hue="Category",  # Hue is set on the grid
            height=7,
            ratio=5,  # Ratio of joint plot size to marginal plot size
            space=0.1,  # Space between joint and marginal plots
        )
        g.plot_joint(sns.scatterplot, palette=palette, s=40, alpha=0.5, legend=False)
        g.plot_marginals(
            sns.histplot,
            palette=palette,
            bins=50,
            multiple="stack",
            alpha=0.7,
            legend=False,
        )
        g.set_axis_labels(r"$\nu^\epsilon$", r"$\nu^r$", fontsize=20)
        plt.suptitle(
            r"$\nu^r$ vs $\nu^\epsilon$ by Feature Category", fontsize=16, y=1.02
        )
        ordered_cats_in_plot = [
            cat for cat in categories_to_display if cat in df_plot["Category"].unique()
        ]
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=palette[cat])
            for cat in ordered_cats_in_plot
            if cat in palette
        ]
        labels = ordered_cats_in_plot
        if handles:
            g.fig.legend(
                handles,
                labels,
                title="Feature Category",
                loc="upper right",
                bbox_to_anchor=(0.99, 0.97),
            )

        # Style adjustments
        g.ax_joint.grid(True, linestyle="-", alpha=0.3, color="gray")
        g.ax_marg_x.grid(True, linestyle="-", alpha=0.3, color="gray")
        g.ax_marg_y.grid(True, linestyle="-", alpha=0.3, color="gray")
        g.ax_joint.tick_params(axis="both", which="major", labelsize=14)

        # g.ax_joint.set_xlim(0, 1.0)
        # g.ax_joint.set_ylim(0, 1.0)

        # Save plot
        if self.save_dir:
            filename = "nu_plot_jointgrid.png"
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, filename)
            try:
                # Use bbox_inches='tight' to prevent labels from being cut off
                g.fig.savefig(save_path, bbox_inches="tight", dpi=150)
                print(f"Nu plot saved to {save_path}")
            except Exception as e:
                print(f"Error saving nu plot: {e}")

        return g.fig

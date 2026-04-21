"""
Module providing the ShaTS abstract class and its explainers
"""
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from .grouping import (
    AbstractGroupingStrategy,
    FeaturesGroupingStrategy,
    MultifeaturesGroupingStrategy,
    TimeGroupingStrategy,
)
from .utils import (
    BackgroundDatasetStrategy,
    StrategySubsets,
    estimate_m,
    generate_subsets,
    infer_binary_feature_indices,
    resolve_background_dataset,
    validate_window_dataset,
    integrated_gradients_groups_direct
)


def _clear_cuda_cache() -> None:
    """
    Clear the CUDA cache when CUDA is available.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class _WeightedLinearRegression(nn.Module):
    """
    Weighted linear regression model used by KernelShaTS.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Run the regression model.
        """
        return self.linear(x)

class ShaTS(ABC):
    """
    Abstract class for initializing ShaTS module

    Args:
        model_wrapper (Callable[..., Tensor]): A function that wraps the model to be explained and returns the output.
        background_dataset (Sequence[Tensor] | None): A dataset to be used as background for the explainer. If None, it will be inferred from the train_dataset.
        train_dataset (Sequence[Tensor] | None): The training dataset, used to infer the background dataset if background_dataset is None.
        background_dataset_strategy (BackgroundDatasetStrategy): The strategy to infer the background dataset if background_dataset is None.
        train_labels (Sequence[int] | Tensor | None): The labels of the training dataset, used to infer the background dataset if background_dataset is None and background_dataset_strategy is not RANDOM.
        grouping_strategy (str | AbstractGroupingStrategy): The strategy for grouping features. 
        subsets_generation_strategy (StrategySubsets): The strategy for generating subsets.
        m (int): Number of subsets to be generated.
        batch_size (int): Size of the batches for processing data.
        device (str | torch.device | int): Device to perform computations on.
        custom_groups (list[list[int]] | None): Custom groups for multifeature grouping strategy.
    
    """
    def __init__(
        self,
        model_wrapper: Callable[[Tensor], Tensor],
        background_dataset: Sequence[Tensor] | None = None,
        train_dataset: Sequence[Tensor] | None = None,
        background_dataset_strategy: BackgroundDatasetStrategy = BackgroundDatasetStrategy.RANDOM,
        background_size: int | None = None,
        train_labels: Sequence[int] | Tensor | None = None,
        grouping_strategy: str | AbstractGroupingStrategy = "time",
        subsets_generation_strategy: StrategySubsets = StrategySubsets.APPROX,
        m: int = 5,
        batch_size: int = 32,
        device: str | torch.device | int = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        custom_groups: list[list[int]] | None = None,
        random_state: int | None = None,
        entropy_bins: int = 32,
        kmeans_max_iter: int = 100,
        kmeans_tolerance: float = 1e-4,
    ) -> None:
        self.device = device
        self.model_wrapper = model_wrapper
        self.batch_size = batch_size
        self.subsets_generation_strategy = subsets_generation_strategy

        self.background_dataset = resolve_background_dataset(
            background_dataset=background_dataset,
            train_dataset=train_dataset,
            background_size=background_size,
            background_dataset_strategy=background_dataset_strategy,
            train_labels=train_labels,
            random_state=random_state,
            entropy_bins=entropy_bins,
            kmeans_max_iter=kmeans_max_iter,
            kmeans_tolerance=kmeans_tolerance,
        )
        validate_window_dataset(self.background_dataset, "background_dataset")

        self.background_tensor = torch.stack(self.background_dataset, dim=0).to(self.device)
        self.window_size = self.background_dataset[0].shape[0]
        self.num_of_features = self.background_dataset[0].shape[1]

        self.grouping_strategy = self._resolve_grouping_strategy(
            grouping_strategy=grouping_strategy,
            custom_groups=custom_groups,
        )

        self.nclass = self._call_model(self.background_dataset[0]).shape[1]
        self.m = estimate_m(self.groups_num, m)

        generated_subsets = generate_subsets(
            self.groups_num,
            self.m,
            self.subsets_generation_strategy,
        )
        self.subsets_dict = generated_subsets.predictors_to_subsets
        self.all_subsets = generated_subsets.all_subsets

        keys_background_subsets = [
            (tuple(subset), entity)
            for subset in self.all_subsets
            for entity in range(len(self.background_dataset))
        ]
        self.pair_dicts = {
            (subset, entity): index
            for index, (subset, entity) in enumerate(keys_background_subsets)
        }

        self.coefficients_dict = self._generate_coefficients_dict()
        self.mean_prediction = self._compute_mean_prediction()

    @property
    def groups_num(self) -> int:
        """
        Return the number of groups defined by the grouping strategy.
        """
        return self.grouping_strategy.groups_num

    @abstractmethod
    def compute(self, test_dataset: Sequence[Tensor]) -> Tensor:
        """
        Compute ShaTS values for the given test dataset.
        """
        raise NotImplementedError

    def plot(
        self,
        shats_values: Tensor,
        test_dataset: Sequence[Tensor] | None = None,
        predictions: Tensor | None = None,
        path: str | Path | None = None,
        segment_size: int = 100,
        class_to_explain: int = 0,
    ) -> None:
        """
        Plot ShaTS values together with the model output.
        """
        if test_dataset is None and predictions is None:
            raise ValueError("Either test_dataset or predictions must be provided.")
        if test_dataset is not None and predictions is not None:
            raise ValueError("Only one of test_dataset or predictions should be provided.")

        if predictions is not None:
            model_predictions = predictions.detach().cpu().numpy()
        else:
            validate_window_dataset(test_dataset, "test_dataset")  # type: ignore[arg-type]
            model_predictions = np.zeros(len(test_dataset), dtype=float)
            for index, data in enumerate(test_dataset or []):
                model_predictions[index] = float(
                    self._call_model(data)[0, class_to_explain].detach().cpu().item()
                )

        shats_values = shats_values[:, :, class_to_explain]
        fontsize = 25
        size = shats_values.shape[0]

        arr_plot = np.zeros((self.groups_num, size), dtype=float)
        arr_prob = np.zeros(size, dtype=float)

        for index in range(size):
            arr_plot[:, index] = shats_values[index].detach().cpu().numpy()
            arr_prob[index] = model_predictions[index]

        vmin, vmax = -0.5, 0.5
        cmap = plt.get_cmap("bwr")

        n_segments = (size + segment_size - 1) // segment_size
        fig, axs = plt.subplots(
            n_segments,
            1,
            figsize=(15, 25 * (max(10, self.groups_num) / 36) * n_segments),
        )

        if n_segments == 1:
            axs = [axs]

        title, y_label, columns_labels = self.grouping_strategy.get_plot_texts()

        for segment_idx in range(n_segments):
            real_end = min((segment_idx + 1) * segment_size, size)

            if segment_idx == n_segments - 1 and size % segment_size != 0:
                pad_size = segment_size - (size % segment_size)
                arr_plot = np.hstack((arr_plot, np.zeros((self.groups_num, pad_size))))
                arr_prob = np.hstack((arr_prob, -np.ones(pad_size)))

            init = segment_idx * segment_size
            end = min((segment_idx + 1) * segment_size, arr_plot.shape[1])
            segment = arr_plot[:, init:end]

            ax = axs[segment_idx]
            ax.set_xlabel("Window", fontsize=fontsize)
            ax.set_ylabel(y_label, fontsize=fontsize)
            ax.set_title(title, fontsize=fontsize)

            cax = ax.imshow(
                segment,
                cmap=cmap,
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )

            cbar_ax = fig.add_axes(
                (
                    ax.get_position().x1 + 0.15,
                    ax.get_position().y0 - 0.05,
                    0.05,
                    ax.get_position().height + 0.125,
                )
            )
            cbar = fig.colorbar(cax, cax=cbar_ax, orientation="vertical")
            cbar.ax.tick_params(labelsize=fontsize)

            ax2 = ax.twinx()
            prediction = arr_prob[init:real_end]
            ax2.plot(
                np.arange(0, real_end - init),
                prediction,
                linestyle="--",
                color="darkviolet",
                linewidth=4,
            )
            ax2.axhline(0.5, color="black", linewidth=1, linestyle="--")
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis="y", labelsize=fontsize)
            ax2.set_ylabel("Model outcome", fontsize=fontsize)

            legend = ax2.legend(
                ["Model outcome", "Threshold"],
                fontsize=fontsize,
                loc="lower left",
                bbox_to_anchor=(0.0, -0.0),
            )
            legend.get_frame().set_alpha(None)
            legend.get_frame().set_facecolor((0, 0, 0, 0))
            legend.get_frame().set_edgecolor("black")

            ax.set_yticks(np.arange(self.groups_num))
            ax.set_yticklabels(columns_labels, fontsize=fontsize)

            xticks = np.arange(0, segment.shape[1], 5)
            xlabels = np.arange(init, real_end, 5)
            xticks = xticks[: len(xlabels)]

            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels, fontsize=fontsize)

        if path is not None:
            plt.savefig(path)
        plt.show()

    def _resolve_grouping_strategy(
        self,
        grouping_strategy: str | AbstractGroupingStrategy,
        custom_groups: list[list[int]] | None,
    ) -> AbstractGroupingStrategy:
        """
        Build the grouping strategy used by the explainer.
        """
        if isinstance(grouping_strategy, AbstractGroupingStrategy):
            return grouping_strategy

        if grouping_strategy == "time":
            return TimeGroupingStrategy(groups_num=self.window_size)
        if grouping_strategy == "feature":
            return FeaturesGroupingStrategy(groups_num=self.num_of_features)
        if grouping_strategy == "multifeature":
            if custom_groups is None:
                raise ValueError(
                    "custom_groups must be provided when grouping_strategy is 'multifeature'."
                )
            return MultifeaturesGroupingStrategy(
                groups_num=len(custom_groups),
                custom_groups=custom_groups,
            )

        raise ValueError(
            "grouping_strategy must be 'time', 'feature', 'multifeature', "
            "or an AbstractGroupingStrategy instance."
        )

    def _call_model(self, data: Tensor) -> Tensor:
        """
        Call the wrapped model and return a tensor of shape [batch, nclass].
        """
        data = data.to(self.device)
        if data.dim() == 2:
            data = data.unsqueeze(0)

        output = self.model_wrapper(data)
        if output.dim() == 1:
            output = output.unsqueeze(0)
        if output.dim() != 2:
            raise ValueError("model_wrapper must return a tensor of shape [batch, nclass].")
        return output

    def _generate_coefficients_dict(self) -> dict[int, float]:
        """
        Generate Shapley coefficients for each coalition size.
        """
        coef_dict: dict[int, float] = {}
        if self.subsets_generation_strategy == StrategySubsets.EXACT:
            for coalition_size in range(self.groups_num):
                coef_dict[coalition_size] = (
                    math.factorial(coalition_size)
                    * math.factorial(self.groups_num - coalition_size - 1)
                    / math.factorial(self.groups_num)
                )
        else:
            for coalition_size in range(self.groups_num):
                coef_dict[coalition_size] = 1 / self.groups_num
        return coef_dict

    def _compute_mean_prediction(self) -> Tensor:
        """
        Compute the mean model prediction over the background dataset.
        """
        mean_prediction = torch.zeros(self.nclass, device=self.device)

        with torch.no_grad():
            for data in self.background_dataset:
                probs = self._call_model(data)[0]
                mean_prediction += probs

        return mean_prediction / len(self.background_dataset)

    def _modify_data_batches(self, data: Tensor) -> list[Tensor]:
        """
        Build the modified batches required by ShaTS for one instance.
        """
        modified_data_batches: list[Tensor] = []

        for subset in self.all_subsets:
            data_tensor = (
                data.unsqueeze(0)
                .expand(len(self.background_dataset), *data.shape)
                .clone()
                .to(self.device)
            )
            modified_data_batches.append(
                self.grouping_strategy.modify_tensor(
                    subset=subset,
                    device=self.device,
                    background_tensor=self.background_tensor,
                    tensor=data_tensor,
                )
            )

        return modified_data_batches

    def _compute_probs(self, modified_data_batches: Sequence[Tensor]) -> list[Tensor]:
        """
        Compute model probabilities for all modified batches.
        """
        probs: list[list[Tensor]] = [[] for _ in range(self.nclass)]

        for batch_start in range(0, len(modified_data_batches), self.batch_size):
            batch = torch.cat(
                modified_data_batches[batch_start : batch_start + self.batch_size],
                dim=0,
            ).to(self.device)
            batch_probs = self._call_model(batch)

            for class_idx in range(self.nclass):
                probs[class_idx].append(batch_probs[:, class_idx])

        return [torch.cat(class_probs, dim=0) for class_probs in probs]

    def _compute_differences(
        self,
        probs: Sequence[Tensor],
        group: int,
        size: int,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute the model output differences for the subsets with and without one group.
        """
        subsets_with, subsets_without = self.subsets_dict[(group, size)]
        prob_with = torch.zeros(self.nclass, len(subsets_with), device=self.device)
        prob_without = torch.zeros(self.nclass, len(subsets_without), device=self.device)

        for pair_index, (item_with, item_without) in enumerate(
            zip(subsets_with, subsets_without)
        ):
            indexes_with = [
                self.pair_dicts[(tuple(item_with), entity)]
                for entity in range(len(self.background_dataset))
            ]
            indexes_without = [
                self.pair_dicts[(tuple(item_without), entity)]
                for entity in range(len(self.background_dataset))
            ]

            indexes_with_tensor = torch.tensor(indexes_with, dtype=torch.long, device=self.device)
            indexes_without_tensor = torch.tensor(
                indexes_without, dtype=torch.long, device=self.device
            )
            coef = self.coefficients_dict[len(item_without)]

            mean_probs_with = torch.zeros(self.nclass, device=self.device)
            mean_probs_without = torch.zeros(self.nclass, device=self.device)

            for class_idx in range(self.nclass):
                selected_probs_with = torch.index_select(
                    probs[class_idx], 0, indexes_with_tensor
                )
                selected_probs_without = torch.index_select(
                    probs[class_idx], 0, indexes_without_tensor
                )

                mean_probs_with[class_idx] = selected_probs_with.mean() * coef
                mean_probs_without[class_idx] = selected_probs_without.mean() * coef

            prob_with[:, pair_index] = mean_probs_with
            prob_without[:, pair_index] = mean_probs_without

        return prob_with, prob_without



class ApproShaTS(ShaTS):
    """
    Original ShaTS implementation optimized for lower memory usage.
    """

    def compute(self, test_dataset: Sequence[Tensor]) -> Tensor:
        """
        Compute ShaTS values with the original algorithm.
        """
        validate_window_dataset(test_dataset, "test_dataset")
        shats_values_list = torch.zeros(
            len(test_dataset),
            self.groups_num,
            self.nclass,
            device=self.device,
        )

        total = len(test_dataset)
        with torch.no_grad():
            for idx, data in enumerate(test_dataset):
                progress = (idx + 1) / total * 100
                print(f"\rProcessing item {idx + 1}/{total} ({progress:.2f}%)", end="")

                shats_values = torch.zeros(
                    self.groups_num,
                    self.nclass,
                    device=self.device,
                )

                modified_data_batches = self._modify_data_batches(data.to(self.device))
                probs = self._compute_probs(modified_data_batches)

                for group in range(self.groups_num):
                    for size in range(self.groups_num):
                        prob_with, prob_without = self._compute_differences(
                            probs=probs,
                            group=group,
                            size=size,
                        )
                        diff = prob_with - prob_without
                        shats_values[group] += diff.mean(dim=1)

                shats_values_list[idx] = shats_values

                del modified_data_batches, probs, shats_values
                _clear_cuda_cache()

        return shats_values_list


class FastShaTS(ShaTS):
    """
    Faster ShaTS implementation based on a precomputed subset reverse index.
    """

    def compute(self, test_dataset: Sequence[Tensor]) -> Tensor:
        """
        Compute ShaTS values with the fast algorithm.
        """
        validate_window_dataset(test_dataset, "test_dataset")
        shats_values_list = torch.zeros(
            len(test_dataset),
            self.groups_num,
            self.nclass,
            device=self.device,
        )
        reversed_dict = self._reverse_dict(self.subsets_dict, self.all_subsets)

        total = len(test_dataset)
        with torch.no_grad():
            for idx, data in enumerate(test_dataset):
                progress = (idx + 1) / total * 100
                print(f"\rProcessing item {idx + 1}/{total} ({progress:.2f}%)", end="")

                shats_values = torch.zeros(
                    self.groups_num,
                    self.nclass,
                    device=self.device,
                )

                modified_data_batches = self._modify_data_batches(data.to(self.device))
                probs = self._compute_probs(modified_data_batches)

                for class_idx in range(self.nclass):
                    for subset_index, references in enumerate(reversed_dict.values()):
                        start = subset_index * len(self.background_dataset)
                        end = (subset_index + 1) * len(self.background_dataset)
                        contribution = probs[class_idx][start:end].mean() / self.groups_num

                        for reference in references:
                            group_key, sign_flag = reference
                            normalizer = len(self.subsets_dict[(group_key[0], group_key[1])][0])

                            if sign_flag == 0:
                                shats_values[group_key[0], class_idx] -= contribution / normalizer
                            else:
                                shats_values[group_key[0], class_idx] += contribution / normalizer

                shats_values_list[idx] = shats_values

                del modified_data_batches, probs, shats_values
                _clear_cuda_cache()

        return shats_values_list

    def _reverse_dict(
        self,
        subsets_dict: dict[Any, Any],
        subsets_total: Sequence[Any],
    ) -> dict[tuple[Any, ...], list[Any]]:
        """
        Build the reverse mapping from subset to group references.
        """
        subsets_dict_reversed: dict[tuple[Any, ...], list[Any]] = {
            tuple(subset): [] for subset in subsets_total
        }

        for subset in subsets_total:
            for key, value in subsets_dict.items():
                if list(subset) in value[0]:
                    subsets_dict_reversed[tuple(subset)].append((key, 0))
                if list(subset) in value[1]:
                    subsets_dict_reversed[tuple(subset)].append((key, 1))

        return subsets_dict_reversed


class KernelShaTS(ShaTS):
    """
    Kernel-based ShaTS implementation.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.binary_vectors = self._subsets_to_binary_vectors()
        self.weights = self._compute_weights()

    def compute(
        self,
        test_dataset: Sequence[Tensor],
        learning_rate: float = 0.01,
        early_stopping_rate: float = 0.1,
        num_epochs: int = 10,
    ) -> Tensor:
        """
        Compute ShaTS values with a weighted linear regression approximation.
        """
        validate_window_dataset(test_dataset, "test_dataset")
        shats_values_list = torch.zeros(
            len(test_dataset),
            self.groups_num,
            self.nclass,
            device=self.device,
        )

        binary_vectors_tensor = torch.tensor(
            self.binary_vectors,
            dtype=torch.float32,
            device=self.device,
        )
        weights_tensor = torch.tensor(
            self.weights,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        for idx, data in enumerate(test_dataset):
            with torch.no_grad():
                modified_data_batches = self._modify_data_batches(data.to(self.device))
                probs = self._compute_probs(modified_data_batches)

            for class_idx in range(self.nclass):
                target = torch.zeros(len(self.all_subsets), 1, device=self.device)
                for subset_index, subset in enumerate(self.all_subsets):
                    indexes = [
                        self.pair_dicts[(tuple(subset), entity)]
                        for entity in range(len(self.background_dataset))
                    ]
                    target[subset_index, 0] = probs[class_idx][indexes].mean()

                regression_model = _WeightedLinearRegression(
                    input_dim=binary_vectors_tensor.shape[1]
                ).to(self.device)
                optimizer = optim.Adam(regression_model.parameters(), lr=learning_rate)

                last_loss: float | None = None
                for _ in range(num_epochs):
                    optimizer.zero_grad()
                    prediction = regression_model(binary_vectors_tensor)
                    loss = torch.sum(weights_tensor * (target - prediction) ** 2)
                    loss.backward()

                    current_loss = float(loss.item())
                    if last_loss is not None:
                        improvement = abs(last_loss - current_loss) / max(abs(last_loss), 1e-12)
                        if improvement < early_stopping_rate:
                            break
                    last_loss = current_loss
                    optimizer.step()

                raw_weights = regression_model.linear.weight.detach().flatten()
                shats_values_list[idx, :, class_idx] = raw_weights

            del modified_data_batches, probs
            _clear_cuda_cache()

        return shats_values_list

    def _subsets_to_binary_vectors(self) -> np.ndarray:
        """
        Convert subsets into binary indicator vectors.
        """
        binary_vectors = np.zeros((len(self.all_subsets), self.groups_num), dtype=int)
        for index, subset in enumerate(self.all_subsets):
            binary_vectors[index, list(subset)] = 1
        return binary_vectors

    def _compute_weights(self) -> list[float]:
        """
        Compute kernel weights for the binary subset vectors.
        """
        weights: list[float] = []
        total_groups = self.groups_num

        for coalition in self.binary_vectors:
            coalition_size = int(np.sum(coalition))
            if coalition_size == 0 or coalition_size == total_groups:
                weights.append(0.0)
            else:
                weight = (total_groups - 1) / (
                    math.comb(total_groups, coalition_size)
                    * coalition_size
                    * (total_groups - coalition_size)
                )
                weights.append(float(weight))

        return weights

##IG Explainer

class FastShaTSIG(FastShaTS):
    """
    FastShaTS variant that updates explanations with group-level Integrated Gradients between close windows.
    """

    def __init__(
        self,
        model_wrapper: Callable[[Tensor], Tensor],
        grad_model_wrapper: Callable[[Tensor], Tensor],
        background_dataset: Sequence[Tensor] | None = None,
        train_dataset: Sequence[Tensor] | None = None,
        background_dataset_strategy: BackgroundDatasetStrategy = BackgroundDatasetStrategy.RANDOM,
        background_size: int | None = None,
        train_labels: Sequence[int] | Tensor | None = None,
        grouping_strategy: str | AbstractGroupingStrategy = "time",
        subsets_generation_strategy: StrategySubsets = StrategySubsets.APPROX,
        m: int = 5,
        batch_size: int = 32,
        device: str | torch.device | int = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        custom_groups: list[list[int]] | None = None,
        random_state: int | None = None,
        entropy_bins: int = 32,
        kmeans_max_iter: int = 100,
        kmeans_tolerance: float = 1e-4,
        distance_threshold: float = 0.05,
        pred_distance_threshold: float | None = None,
        recalculation_mode: str = "x",
        force_full_every: int = 100,
        ig_steps: int = 32,
        ig_lambda: float = 1.0,
        ig_class_idx: int = 1,
        categorical_feature_indices: Sequence[int] | None = None,
        detect_categorical_features: bool = True,
    ) -> None:
        super().__init__(
            model_wrapper=model_wrapper,
            background_dataset=background_dataset,
            train_dataset=train_dataset,
            background_dataset_strategy=background_dataset_strategy,
            background_size=background_size,
            train_labels=train_labels,
            grouping_strategy=grouping_strategy,
            subsets_generation_strategy=subsets_generation_strategy,
            m=m,
            batch_size=batch_size,
            device=device,
            custom_groups=custom_groups,
            random_state=random_state,
            entropy_bins=entropy_bins,
            kmeans_max_iter=kmeans_max_iter,
            kmeans_tolerance=kmeans_tolerance,
        )

        self.grad_model_wrapper = grad_model_wrapper
        self.distance_threshold = float(distance_threshold)
        self.pred_distance_threshold = (
            None if pred_distance_threshold is None else float(pred_distance_threshold)
        )
        self.recalculation_mode = recalculation_mode
        self.force_full_every = force_full_every
        self.ig_steps = ig_steps
        self.ig_lambda = ig_lambda
        self.ig_class_idx = ig_class_idx

        if categorical_feature_indices is not None:
            self.categorical_feature_indices = sorted(set(int(index) for index in categorical_feature_indices))
        elif detect_categorical_features:
            self.categorical_feature_indices = infer_binary_feature_indices(
                self.background_dataset
            )
        else:
            self.categorical_feature_indices = []

    def _input_distance(self, x_new: Tensor, x_old: Tensor) -> float:
        """
        Compute the relative input distance between two windows.
        """
        diff = x_new - x_old
        numerator = torch.norm(diff)
        denominator = torch.norm(x_old) + 1e-8
        return float((numerator / denominator).item())

    def _pred_distance(self, x_prev: Tensor, x_curr: Tensor) -> float:
        """
        Compute the L1 distance between the model predictions of two windows.
        """
        with torch.no_grad():
            p_prev = self._call_model(x_prev)
            p_curr = self._call_model(x_curr)
        return float(torch.abs(p_curr - p_prev).sum().item())

    def _integrated_gradients_groups(self, x_start: Tensor, x_end: Tensor) -> Tensor:
        """
        Compute group-level Integrated Gradients between two windows.
        """
        return integrated_gradients_groups_direct(
            model_wrapper=self.grad_model_wrapper,
            x=x_end,
            baseline=x_start,
            grouping_strategy=self.grouping_strategy,
            class_idx=self.ig_class_idx,
            steps=self.ig_steps,
            device=self.device,
            categorical_feature_indices=self.categorical_feature_indices,
        )

    def _compute_full_shap_single(self, x: Tensor) -> Tensor:
        """
        Compute a full FastShaTS explanation for one window.
        """
        with torch.no_grad():
            full = super().compute([x])
        return full[0]

    def compute(
        self,
        test_dataset: Sequence[Tensor],
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, list[float], list[bool]]:
        """
        Compute FastShaTSIG values for the given test dataset.
        """
        validate_window_dataset(test_dataset, "test_dataset")
        num_instances = len(test_dataset)
        shats_values = torch.zeros(
            num_instances,
            self.groups_num,
            self.nclass,
            device=self.device,
        )

        prev_x: Tensor | None = None
        prev_shap: Tensor | None = None

        times: list[float] = [0.0] * num_instances
        approx_flags: list[bool] = [False] * num_instances

        full_recomputes = 0
        ig_updates = 0

        global_start = time.perf_counter()
        ema: float | None = None

        for idx, x in enumerate(test_dataset):
            x = x.to(self.device)
            step_start = time.perf_counter()

            need_full = False
            if idx == 0:
                need_full = True
            elif self.force_full_every > 0 and idx % self.force_full_every == 0:
                need_full = True
            elif prev_x is None or prev_shap is None:
                need_full = True
            else:
                distance_x = self._input_distance(x, prev_x)
                distance_p = (
                    self._pred_distance(prev_x, x)
                    if self.pred_distance_threshold is not None
                    else None
                )

                if self.recalculation_mode == "x":
                    need_full = distance_x > self.distance_threshold
                elif self.recalculation_mode == "pred":
                    if self.pred_distance_threshold is None or distance_p is None:
                        need_full = distance_x > self.distance_threshold
                    else:
                        need_full = distance_p > self.pred_distance_threshold
                elif self.recalculation_mode == "x_or_pred":
                    if self.pred_distance_threshold is None or distance_p is None:
                        need_full = distance_x > self.distance_threshold
                    else:
                        need_full = (
                            distance_x > self.distance_threshold
                            or distance_p > self.pred_distance_threshold
                        )
                else:
                    raise ValueError(
                        f"Unknown recalculation_mode={self.recalculation_mode!r}."
                    )

            if need_full:
                full_recomputes += 1
                shap_t = self._compute_full_shap_single(x)
                prev_x, prev_shap = x, shap_t
                approx_flags[idx] = False
            else:
                ig_updates += 1
                group_ig = self._integrated_gradients_groups(prev_x, x)
                shap_t = prev_shap.clone()
                shap_t[:, self.ig_class_idx] = (
                    shap_t[:, self.ig_class_idx] + self.ig_lambda * group_ig
                )
                prev_x, prev_shap = x, shap_t
                approx_flags[idx] = True

            shats_values[idx] = shap_t

            step_end = time.perf_counter()
            times[idx] = step_end - step_start

            current_step = times[idx]
            ema = current_step if ema is None else (0.1 * current_step + 0.9 * ema)
            if ((idx + 1) % 50 == 0) or (idx + 1 == num_instances):
                elapsed = step_end - global_start
                eta = ema * (num_instances - (idx + 1))
                print(
                    f"\rFastShaTSIG - Processing {idx + 1}/{num_instances} "
                    f"({(idx + 1) / num_instances * 100.0:.2f}%) | "
                    f"full={full_recomputes} | ig={ig_updates} | "
                    f"elapsed={elapsed:.1f}s | ETA={eta:.1f}s",
                    end="",
                    flush=True,
                )

        print()

        if return_diagnostics:
            return shats_values, times, approx_flags
        return shats_values
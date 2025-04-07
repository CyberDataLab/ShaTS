import math
from pathlib import Path
from typing import Any
from typing import Callable
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from tsg_shap.grouping import AbstractGroupingStrategy, TimeGroupingStrategy, FeaturesGroupingStrategy, MultifeaturesGroupingStrategy

from .utils import StrategySubsets, estimate_m, generate_subsets


class ShaTS(ABC):
    "Abstract class for initializing ShaTS module"
    def __init__(
        self,
        model: Module,
        model_wrapper: Callable[..., Tensor],
        support_dataset: list[Tensor],
        grouping_strategy: str | AbstractGroupingStrategy,
        subsets_generation_strategy: StrategySubsets = StrategySubsets.APPROX,
        m: int = 5,
        batch_size: int = 32,
        device: str | torch.device | int = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        custom_groups: list[list[int]] | None = None,
    ):

        self.model = model
        self.support_dataset = support_dataset
        self.support_tensor = torch.stack(
            [data for data in support_dataset]
        ).to(device)
        self.window_size = support_dataset[0].shape[0]
        self.num_of_features = support_dataset[0].shape[1]
        self.subsets_generation_strategy = subsets_generation_strategy
        self.grouping_strategy : AbstractGroupingStrategy
        
        if isinstance(grouping_strategy, AbstractGroupingStrategy):
            self.grouping_strategy = grouping_strategy

        elif grouping_strategy == "time":
            self.grouping_strategy = TimeGroupingStrategy(
                groups_num=self.window_size
            )
        elif grouping_strategy == "feature":
            self.grouping_strategy = FeaturesGroupingStrategy(
                groups_num=self.num_of_features
            )
        elif grouping_strategy == "multifeature":
            if custom_groups is None:
                raise ValueError(
                    "custom_groups must be provided when grouping_strategy is 'multifeature'."
                )
            self.grouping_strategy = MultifeaturesGroupingStrategy(
                groups_num=len(custom_groups),
                custom_groups=custom_groups,
            )
        else:
            raise ValueError(
                "grouping_strategy must be 'time', 'feature', or 'multifeature'."
            )

        
        self.device = device
        self.batch_size = batch_size
        self.nclass = model_wrapper(model, support_dataset[0]).shape[1]
        self.model_wrapper = model_wrapper

        self.m = estimate_m(self.groups_num, m)

        self.subsets_dict, self.all_subsets = generate_subsets(
            self.groups_num, self.m, self.subsets_generation_strategy
        )

        keys_support_subsets = [
            (tuple(subset), entity)
            for subset in self.all_subsets
            for entity in range(len(self.support_dataset))
        ]
        self.pair_dicts = {
            (subset, entity): i
            for i, (subset, entity) in enumerate(keys_support_subsets)
        }

        self.coefficients_dict = self._generate_coefficients_dict()
        self.mean_prediction = self._compute_mean_prediction()

    @property
    def groups_num(self) -> int:
        return self.grouping_strategy.groups_num
    
    @abstractmethod
    def compute(
        self, 
        test_dataset: list[Tensor]
    ) -> Tensor:
        raise NotImplementedError()

    def plot(
        self,
        shats_values: Tensor,
        test_dataset: list[Tensor] | None = None,
        predictions: Tensor | None = None,
        path: str | Path | None = None,
        segment_size: int = 100,
        class_to_explain: int = 0,
    ):
        if test_dataset is None and predictions is None:
            raise ValueError(
                "Either test_dataset or predictions must be provided."
            )
        if test_dataset is not None and predictions is not None:
            raise ValueError(
                "Only one of test_dataset or predictions should be provided."
            )
        elif predictions is not None:
            model_predictions = predictions
        elif test_dataset is not None:
            model_predictions = torch.zeros(
                len(test_dataset), device=self.device
            )
            for i, data in enumerate(test_dataset):
                model_predictions[i] = self.model_wrapper(self.model, data)[0][class_to_explain]
        
        shats_values = shats_values[:,:, class_to_explain]
        fontsize = 25
        size = shats_values.shape[0]

        arr_plot = np.zeros((self.groups_num, size))
        arr_prob = np.zeros(size)

        for i in range(size):
            arr_plot[:, i] = shats_values[i].cpu().numpy()
            arr_prob[i] = model_predictions[i]

        vmin, vmax = -0.5, 0.5
        cmap = plt.get_cmap("bwr")

        n_segments = (size + segment_size - 1) // segment_size
        fig, axs = plt.subplots(
            n_segments, 1, figsize=(15, 25 * (max(10, self.groups_num) / 36) * n_segments)
        )  # 15, 25 predictor

        if n_segments == 1:
            axs = [axs]

        for n in range(n_segments):
            real_end = min((n + 1) * segment_size, size)
            if n == n_segments - 1:
                real_end = arr_plot.shape[1]
                arr_plot = np.hstack(
                    (
                        arr_plot,
                        np.zeros(
                            (self.groups_num, segment_size - (size % segment_size))
                        ),
                    )
                )
                arr_prob = np.hstack(
                    (arr_prob, -np.ones(segment_size - (size % segment_size)))
                )
                size = arr_plot.shape[1]

            init = n * segment_size
            end = min((n + 1) * segment_size, size)
            segment = arr_plot[:, init:end]
            ax = axs[n]

            ax.set_xlabel("Window", fontsize=fontsize)

            cax = ax.imshow(
                segment,
                cmap=cmap,
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )

            cbar_ax = fig.add_axes(
                [
                    ax.get_position().x1 + 0.15,
                    ax.get_position().y0 - 0.05,
                    0.05,
                    ax.get_position().height + 0.125,
                ]
            )

            cbar = fig.colorbar(cax, cax=cbar_ax, orientation="vertical")
            cbar.ax.tick_params(labelsize=fontsize)

            ax2 = ax.twinx()

            prediction = arr_prob[init:real_end]  # Ajustar a realEnd
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

            title, y_label, columns_labels = self.grouping_strategy.get_plot_texts()

            ax.set_ylabel(y_label, fontsize=fontsize)
            ax.set_title(title, fontsize=fontsize)

            ax.set_yticks(np.arange(self.groups_num))
            ax.set_yticklabels(columns_labels, fontsize=fontsize)

            xticks = np.arange(0, segment.shape[1], 5)
            xlabels = np.arange(init, real_end, 5)

            xticks = xticks[: len(xlabels)]

            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels, fontsize=fontsize)

        plt.tight_layout()

        if path is not None:
            plt.savefig(path)
        plt.show()



    def _generate_coefficients_dict(self) -> dict[int, float]:
        coef_dict = dict[int, float]()
        if self.subsets_generation_strategy.value == StrategySubsets.EXACT.value:
            for i in range(self.groups_num):
                coef_dict[i] = (
                    math.factorial(i)
                    * math.factorial(self.groups_num - i - 1)
                    / math.factorial(self.groups_num)
                )
        else:
            for i in range(self.groups_num):
                coef_dict[i] = 1 / self.groups_num
        return coef_dict

    def _compute_mean_prediction(self) -> Tensor:
        mean_prediction = torch.zeros(self.nclass, device=self.device)

        with torch.no_grad():
            for data in self.support_dataset:
                probs = self.model_wrapper(self.model, data)
                for class_idx in range(self.nclass):
                    mean_prediction[class_idx] += probs[0, class_idx].cpu()

        return mean_prediction / len(self.support_dataset)


    def _modify_data_batches(self, data: Tensor) -> list[Tensor]:
        modified_data_batches = list[Tensor]()

        for subset in self.all_subsets:
            data_tensor = (
                data
                .unsqueeze(0)
                .expand(len(self.support_dataset), *data.shape)
                .clone()
                .to(self.device)
            )
            modified_data_batches.append(
                self.grouping_strategy.modify_tensor(
                    subset, self.device, self.support_tensor, data_tensor
                )
            )

        return modified_data_batches

    def _compute_probs(
        self, 
        modified_data_batches: list[Tensor]
    ) -> list[Tensor]:
        probs: list[list[Tensor]] = []
        probs = [[] for _ in range(self.nclass)]

        for i in range(0, len(modified_data_batches), self.batch_size):
            batch = torch.cat(modified_data_batches[i : i + self.batch_size]).to(self.device)
            
            batch_probs = self.model_wrapper(self.model, batch)

            for class_idx in range(self.nclass):
                class_probs = batch_probs[:, class_idx].cpu()
                probs[class_idx].append(class_probs)

        probs = [torch.cat(class_probs, dim=0).to(self.device) for class_probs in probs]

        return probs

    def _compute_differences(
        self, probs: Tensor, instant: int, size: int
    ) -> tuple[Tensor, Tensor]:
        subsets_with, subsets_without = self.subsets_dict[(instant, size)]
        prob_with = torch.zeros(self.nclass, len(subsets_with), device=self.device)
        prob_without = torch.zeros(self.nclass, len(subsets_without), device=self.device)

        for i, (item_with, item_without) in enumerate(
            zip(subsets_with, subsets_without)
        ):
            indexes_with = [
                self.pair_dicts[(tuple(item_with), entity)]
                for entity in range(len(self.support_dataset))
            ]
            indexes_without = [
                self.pair_dicts[(tuple(item_without), entity)]
                for entity in range(len(self.support_dataset))
            ]
            # Convert indexes to tensors
            indexes_with = torch.tensor(indexes_with, dtype=torch.long, device=self.device)
            indexes_without = torch.tensor(indexes_without, dtype=torch.long, device=self.device)

            coef = self.coefficients_dict[len(item_without)]

            # Initialize tensors for storing the selected probabilities
            mean_probs_with = torch.zeros(self.nclass, device=self.device)
            mean_probs_without = torch.zeros(self.nclass, device=self.device)

            # Iterate over each class and compute the mean probability
            for class_idx in range(self.nclass):
                # Select probabilities for the current class
                selected_probs_with = torch.index_select(probs[class_idx], 0, indexes_with)
                selected_probs_without = torch.index_select(probs[class_idx], 0, indexes_without)

                # Compute the mean of selected probabilities for each class
                mean_probs_with[class_idx] = selected_probs_with.mean() * coef
                mean_probs_without[class_idx] = selected_probs_without.mean() * coef

            # Assign to the probability tensors
            prob_with[:, i] = mean_probs_with
            prob_without[:, i] = mean_probs_without


        return prob_with, prob_without
    



class ApproShaTS(ShaTS):
    def __init__(
        self,
        model: Module,
        model_wrapper: Callable[..., Tensor],
        support_dataset: list[Tensor],
        grouping_strategy: str | AbstractGroupingStrategy,
        subsets_generation_strategy: StrategySubsets = StrategySubsets.APPROX,
        m: int = 5,
        batch_size: int = 32,
        device: str | torch.device | int = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        custom_groups: list[list[int]] | None = None,
    ):
        super().__init__(
            model=model,
            support_dataset=support_dataset,
            grouping_strategy=grouping_strategy,
            subsets_generation_strategy=subsets_generation_strategy,
            m=m,
            batch_size=batch_size,
            device=device,
            custom_groups=custom_groups,
            model_wrapper=model_wrapper
        )

    def compute(
        self,
        test_dataset: list[Tensor]
    ) -> Tensor:
        shats_values_list = torch.zeros(
        len(test_dataset), 
        self.groups_num, 
        self.nclass,
        device=self.device
        )
        total = len(test_dataset)
        with torch.no_grad():
            for idx, data in enumerate(test_dataset):
                progress = (idx + 1) / total * 100
                print(f"\rProcessing item {idx + 1}/{total} ({progress:.2f}%)", end="")

                tsgshapvalues = torch.zeros(self.groups_num, self.nclass, device=self.device)

                modified_data_batches = self._modify_data_batches(data)
                probs = self._compute_probs(modified_data_batches)

                for group in range(self.groups_num):
                    for size in range(self.groups_num):
                        prob_with, prob_without = self._compute_differences(
                            probs, group, size
                        )

                        resta = prob_with - prob_without

                        for class_idx in range(self.nclass):
                            tsgshapvalues[group, class_idx] += resta[class_idx].mean()

                shats_values_list[idx] = tsgshapvalues.clone()

                del (
                    modified_data_batches,
                    probs,
                    tsgshapvalues,
                )
                torch.cuda.empty_cache()

        return shats_values_list




class FastShaTS(ShaTS):
    def __init__(
        self,
        model: Module,
        model_wrapper: Callable[..., Tensor],
        support_dataset: list[Tensor],
        grouping_strategy: str | AbstractGroupingStrategy,
        subsets_generation_strategy: StrategySubsets = StrategySubsets.APPROX,
        m: int = 5,
        batch_size: int = 32,
        device: str | torch.device | int = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        custom_groups: list[list[int]] | None = None,
    ):
        super().__init__(
            model=model,
            support_dataset=support_dataset,
            grouping_strategy=grouping_strategy,
            subsets_generation_strategy=subsets_generation_strategy,
            m=m,
            batch_size=batch_size,
            device=device,
            custom_groups=custom_groups,
            model_wrapper=model_wrapper
        )

    def compute(
        self,
        test_dataset: list[Tensor]
    ) -> Tensor:
        tsgshapvalues_list = torch.zeros(
        len(test_dataset), 
        self.groups_num, 
        self.nclass,
        device=self.device
        )
        reversed_dict = self._reverse_dict(self.subsets_dict, self.all_subsets)

        total = len(test_dataset)
        with torch.no_grad():
            for idx, data in enumerate(test_dataset):
                progress = (idx + 1) / total * 100
                print(f"\rProcessing item {idx + 1}/{total} ({progress:.2f}%)", end="")

                tsgshapvalues = torch.zeros(self.groups_num, self.nclass, device=self.device)

                modified_data_batches = self._modify_data_batches(data)

                probs = self._compute_probs(modified_data_batches)

                for class_idx in range(self.nclass):

                    for i, value in enumerate(reversed_dict.values()):
                        add = probs[class_idx][
                            i
                            * len(self.support_dataset) : (i + 1)
                            * len(self.support_dataset)
                        ].mean()

                        add = add / self.groups_num
                        for v in value:

                            if v[1] == 0:
                                tsgshapvalues[v[0][0]][class_idx] -= add / len(
                                    self.subsets_dict[(v[0][0], v[0][1])][0]
                                )

                            else:
                                tsgshapvalues[v[0][0]][class_idx] += add / len(
                                    self.subsets_dict[(v[0][0], v[0][1])][0]
                                )
                    tsgshapvalues_list[idx] = tsgshapvalues.clone()

                del (
                    modified_data_batches,
                    probs,
                    tsgshapvalues,
                )

        return tsgshapvalues_list
    

    def _reverse_dict(
        self, 
        subsets_dict: dict[Any, Any], 
        subsets_total: list[Any]
    ) -> dict[tuple[Any], list[Any]]:
        subsets_dict_reversed = dict[tuple[Any], list[Any]]()
        for subset in subsets_total:
            subsets_dict_reversed[tuple(subset)] = []

        for subset in subsets_total:
            for clave, valor in subsets_dict.items():
                if list(subset) in valor[0]:
                    subsets_dict_reversed[tuple(subset)].append((clave, 0))

                if list(subset) in valor[1]:
                    subsets_dict_reversed[tuple(subset)].append((clave, 1))

        return subsets_dict_reversed
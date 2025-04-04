import math
from pathlib import Path
from typing import Any
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from tsg_shap.grouping import AbstractGroupingStrategy, TimeGroupingStrategy, FeaturesGroupingStrategy, MultifeaturesGroupingStrategy

from .utils import StrategyPrediction, StrategySubsets, estimate_m, generate_subsets


class ShaTS:
    def __init__(
        self,
        model: Module,
        support_dataset: list[Tensor],
        grouping_strategy: str | AbstractGroupingStrategy,
        subsets_generation_strategy: StrategySubsets = StrategySubsets.APPROX,
        prediction_strategy: StrategyPrediction = StrategyPrediction.MULTICLASS,
        m: int = 5,
        batch_size: int = 32,
        device: str | torch.device | int = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        nclass: int = 2,
        class_to_explain: int = -1,
        custom_groups: list[list[int]] | None = None,
        model_wrapper: Callable[[Module, Tensor, Tensor], Tensor] | None = None,
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

        
        self.prediction_strategy = prediction_strategy
        self.device = device
        self.batch_size = batch_size
        self.nclass = nclass
        self.class_to_explain = class_to_explain
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
            for i in range(0, len(self.support_dataset), self.batch_size):
                batch = self.support_dataset[i : i + self.batch_size]
                #batch_tensor = torch.stack(batch).to(self.device)
                batch_tensor = torch.stack([data for data in batch]).to(
                    self.device
                )
                mean_prediction += (
                    torch.sum(torch.softmax(self.model(batch_tensor), dim=1), dim=0)
                    if self.prediction_strategy.value
                    == StrategyPrediction.MULTICLASS.value
                    else torch.sum(torch.sigmoid(self.model(batch_tensor)), dim=0)
                )
        return mean_prediction / len(self.support_dataset)

    def compute_tsgshap(self, test_dataset: list[Tensor]):
        shats_values_list = torch.zeros(
            len(test_dataset), self.groups_num, device=self.device
        )
        total = len(test_dataset)
        with torch.no_grad():
            for idx, data in enumerate(test_dataset):
                progress = (idx + 1) / total * 100
                print(f"\rProcessing item {idx + 1}/{total} ({progress:.2f}%)", end="")

                tsgshapvalues = torch.zeros(self.groups_num, device=self.device)

                original_pred, original_class, original_prob = self._predict(data)

                modified_data_batches = self._modify_data_batches(data)
                probs = self._compute_probs(modified_data_batches, original_class)

                for group in range(self.groups_num):
                    for size in range(self.groups_num):
                        prob_with, prob_without = self._compute_differences(
                            probs, group, size
                        )
                        tsgshapvalues[group] += (prob_without - prob_with).mean()

                shats_values_list[idx] = tsgshapvalues.clone()

                del (
                    modified_data_batches,
                    probs,
                    original_pred,
                    original_class,
                    original_prob,
                    tsgshapvalues,
                )
                torch.cuda.empty_cache()

        return shats_values_list

    def _predict(self, data: Tensor) -> tuple[Tensor, Tensor | int, Tensor]:
        original_pred = self.model(data.unsqueeze(0).to(self.device))
        original_class = (
            torch.argmax(original_pred)
            if self.prediction_strategy.value == StrategyPrediction.MULTICLASS.value
            else 0
        )
        if self.class_to_explain != -1:
            original_class = self.class_to_explain
        original_prob = (
            torch.softmax(original_pred, dim=1)[0][original_class]
            if self.prediction_strategy.value == StrategyPrediction.MULTICLASS.value
            else torch.sigmoid(original_pred)[0][0]
        )

        return original_pred, original_class, original_prob

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
        self, modified_data_batches: list[Tensor], original_class: Tensor | int
    ) -> Tensor:
        probs: list[Tensor] = []
        for i in range(0, len(modified_data_batches), self.batch_size):
            batch = torch.cat(modified_data_batches[i : i + self.batch_size]).to(
                self.device
            )
            if self.model_wrapper is not None:
                batch_probs = self.model_wrapper(
                    self.model, batch, original_class
                )
            else:
                guesses = self.model(batch)

                batch_probs = (
                    torch.softmax(guesses, dim=1)[:, original_class]
                    if self.prediction_strategy.value == StrategyPrediction.MULTICLASS.value
                    else torch.sigmoid(guesses)[:, 0]
                )


            probs.extend(batch_probs.cpu())

        return torch.tensor(probs, device=self.device)

    def _compute_differences(
        self, probs: Tensor, instant: int, size: int
    ) -> tuple[Tensor, Tensor]:
        subsets_with, subsets_without = self.subsets_dict[(instant, size)]
        prob_with = torch.zeros(len(subsets_with), device=self.device)
        prob_without = torch.zeros(len(subsets_without), device=self.device)

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
            coef = self.coefficients_dict[len(item_without)]
            prob_with[i] = probs[indexes_with].mean() * coef
            prob_without[i] = probs[indexes_without].mean() * coef

        return prob_with, prob_without

    def compute_fast_shats(self, test_dataset: list[Tensor]) -> Tensor:
        tsgshapvalues_list = torch.zeros(
            len(test_dataset), self.groups_num, device=self.device
        )
        reversed_dict = self._reverse_dict(self.subsets_dict, self.all_subsets)

        total = len(test_dataset)
        with torch.no_grad():
            for idx, data in enumerate(test_dataset):
                progress = (idx + 1) / total * 100
                print(f"\rProcessing item {idx + 1}/{total} ({progress:.2f}%)", end="")

                tsgshapvalues = torch.zeros(self.groups_num, device=self.device)

                pred_original, class_original, prob_original = self._predict(data)

                modified_data_batches = self._modify_data_batches(data)

                probs = self._compute_probs(modified_data_batches, class_original)

                for i, value in enumerate(reversed_dict.values()):
                    add = probs[
                        i
                        * len(self.support_dataset) : (i + 1)
                        * len(self.support_dataset)
                    ].mean()

                    add = add / self.groups_num
                    for v in value:

                        if v[1] == 0:
                            tsgshapvalues[v[0][0]] -= add / len(
                                self.subsets_dict[(v[0][0], v[0][1])][0]
                            )

                        else:
                            tsgshapvalues[v[0][0]] += add / len(
                                self.subsets_dict[(v[0][0], v[0][1])][0]
                            )
                tsgshapvalues_list[idx] = tsgshapvalues.clone()

                del (
                    modified_data_batches,
                    probs,
                    pred_original,
                    class_original,
                    prob_original,
                    tsgshapvalues,
                )

        return tsgshapvalues_list

    def _reverse_dict(
        self, subsets_dict: dict[Any, Any], subsets_total: list[Any]
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

    def plot_tsgshap(
        self,
        shats_values: Tensor,
        test_dataset: list[Tensor] | None = None,
        predictions: list[Tensor] | None = None,
        path: str | Path | None = None,
        segment_size: int = 100,
    ):
        if test_dataset is None and predictions is None:
            raise ValueError(
                "Either test_dataset or predictions must be provided."
            )
        if test_dataset is not None and predictions is not None:
            raise ValueError(
                "Only one of test_dataset or predictions should be provided."
            )
        if predictions is not None:
            model_predictions = predictions
        else:
            model_predictions = [
                self._predict(data) for data in test_dataset
            ]

        fontsize = 25
        size = shats_values.shape[0]

        arr_plot = np.zeros((self.groups_num, size))
        arr_prob = np.zeros(size)

        for i in range(size):
            arr_plot[:, i] = shats_values[i].cpu().numpy()
            arr_prob[i] = model_predictions[i][2].detach().cpu().numpy()

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

            # prediction = arr_prob[init:end]

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

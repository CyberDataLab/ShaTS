from abc import ABC, abstractmethod
from typing import NamedTuple

import torch
from torch import Tensor
from torch import device as TorchDevice

from tsg_shap.utils import Subset


class GroupingPlotTexts(NamedTuple):
    title: str
    y_label: str
    columns: list[str]


class AbstractGroupingStrategy(ABC):
    _default_names_key: str

    _plot_title: str
    _plot_y_label: str

    def __init__(self, groups_num: int, names: list[str] | None = None) -> None:
        self.groups_num = groups_num
        self._names = names

    @abstractmethod
    def modify_tensor(
        self,
        subset: Subset,
        device: str | int | TorchDevice,
        support_tensor: Tensor,
        tensor: Tensor,
    ) -> Tensor:
        raise NotImplementedError()

    def get_plot_texts(self) -> GroupingPlotTexts:
        names = self._names or [
            f"{self._default_names_key}{i+1}" for i in range(self.groups_num)
        ]

        return GroupingPlotTexts(self._plot_title, self._plot_y_label, names)


class TimeGroupingStrategy(AbstractGroupingStrategy):
    _default_names_key = "instant"

    _plot_title = "TSG-SHAP (Temporal)"
    _plot_y_label = "Time"

    def modify_tensor(
        self,
        subset: Subset,
        device: str | int | TorchDevice,
        support_tensor: Tensor,
        tensor: Tensor,
    ) -> Tensor:
        indexes = torch.tensor(list(subset), dtype=torch.long, device=device)
        tensor[:, indexes, :] = support_tensor[:, indexes, :].clone()
        return tensor.clone()


class FeaturesGroupingStrategy(AbstractGroupingStrategy):
    _default_names_key = "feature"

    _plot_title = "TSG-SHAP (Feature)"
    _plot_y_label = "Feature"

    def modify_tensor(
        self,
        subset: Subset,
        device: str | int | TorchDevice,
        support_tensor: Tensor,
        tensor: Tensor,
    ) -> Tensor:
        indexes = torch.tensor(list(subset), dtype=torch.long, device=device)
        for instant in range(support_tensor[0].shape[0]):
            tensor[:, instant, indexes] = support_tensor[:, instant, indexes].clone()
        return tensor.clone()


class MultifeaturesGroupingStrategy(AbstractGroupingStrategy):
    _default_names_key = "multifeature"

    _plot_title = "TSG-SHAP ---MULTIFEATURE)"
    _plot_y_label = "MULTIFEATURE"

    def __init__(
        self,
        groups_num: int,
        custom_groups: list[list[int]],
        names: list[str] | None = None,
    ) -> None:
        super().__init__(groups_num, names)

        self._custom_groups = custom_groups

    def modify_tensor(
        self,
        subset: Subset,
        device: str | int | TorchDevice,
        support_tensor: Tensor,
        tensor: Tensor,
    ) -> Tensor:
        all_indexes = list[int]()

        for group in subset:
            all_indexes.extend(self._custom_groups[group])
        all_indexes = torch.tensor(all_indexes, dtype=torch.long, device=device)

        tensor[:, :, all_indexes] = support_tensor[:, :, all_indexes].clone()

        return tensor

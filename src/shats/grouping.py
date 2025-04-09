"""
Module for grouping strategies in ShaTS.
This module contains the abstract class AbstractGroupingStrategy and its
concrete implementations: TimeGroupingStrategy, FeaturesGroupingStrategy,
MultifeaturesGroupingStrategy.

The AbstractGroupingStrategy class defines the interface for grouping
strategies, including the modify_tensor method and the get_plot_texts
method. 

The TimeGroupingStrategy class implements a grouping strategy based on
time, modifying the tensor based on the time dimension. It creates one
group for each time step within the support tensor. Each group is 
conformed by all the features at that time step.

The FeaturesGroupingStrategy class implements a grouping strategy based
on features, modifying the tensor based on the feature dimension. It
creates one group for each feature within the support tensor. Each group
is conformed by all the time steps at that feature.

The MultifeaturesGroupingStrategy class implements a grouping strategy
based on multiple features, modifying the tensor based on a custom
grouping of features. It allows for a more flexible grouping strategy
by allowing the user to define custom groups of features. Each group
is conformed by all the time steps of every feature in the group.
"""

from abc import ABC, abstractmethod
from typing import NamedTuple

import torch
from torch import Tensor
from torch import device as TorchDevice

from .utils import Subset


class GroupingPlotTexts(NamedTuple):
    """
    Named tuple to store the plot texts for grouping strategies.
    """
    title: str
    y_label: str
    columns: list[str]


class AbstractGroupingStrategy(ABC):
    """
    Abstract class for grouping strategies in ShaTS.
    This class defines the interface for grouping strategies, including
    the modify_tensor method and the get_plot_texts method.
    """
    _default_names_key: str

    _plot_title: str
    _plot_y_label: str

    def __init__(self, groups_num: int | None = None, names: list[str] | None = None) -> None:
        if groups_num is None and names is None:
            raise ValueError("groups_num or names must be provided")
        if groups_num is not None and names is not None:
            if groups_num != len(names):
                raise ValueError(
                    "If groups_num and names are provided, they must match in length"
                )
        if groups_num is None and names is not None:
            groups_num = len(names)
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
        """
        Modify the tensor based on the grouping strategy.
        Args:
            subset (Subset): The subset of groups to modify.
            device (str | int | TorchDevice): The device to use for the tensor.
            support_tensor (Tensor): The support tensor to modify.
            tensor (Tensor): The tensor to modify.
        
        Returns:
            Tensor: The modified tensor.
        """
        raise NotImplementedError()

    def get_plot_texts(self) -> GroupingPlotTexts:
        """
        Get the plot texts for the grouping strategy.
        Returns:
            GroupingPlotTexts: A named tuple containing the plot title,
            y_label, and column names.
        """
        names = self._names or [
            f"{self._default_names_key}{i+1}" for i in range(self.groups_num)
        ]

        return GroupingPlotTexts(self._plot_title, self._plot_y_label, names)


class TimeGroupingStrategy(AbstractGroupingStrategy):
    """
    Grouping strategy based on time.
    It creates one group for each time step within every window
    """

    _default_names_key = "instant"

    _plot_title = "ShaTS (Temporal)"
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
    """
    Grouping strategy based on features.
    It creates one group for each feature within the every window.
    """
    _default_names_key = "feature"

    _plot_title = "ShaTS(Feature)"
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
    """
    Grouping strategy based on multiple features.
    It allows for a more flexible grouping strategy by allowing the user
    to define custom groups of features.
    Each group is conformed by all the time steps of every feature in the group.
    """
    _default_names_key = "multifeature"

    _plot_title = "ShaTS (MULTIFEATURE)"
    _plot_y_label = "MULTIFEATURE"

    def __init__(
        self,
        custom_groups: list[list[int]],
        groups_num: int | None = None,
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
        indexes_tensor = torch.tensor(all_indexes, dtype=torch.long, device=device)

        tensor[:, :, indexes_tensor] = support_tensor[:, :, all_indexes].clone()

        return tensor

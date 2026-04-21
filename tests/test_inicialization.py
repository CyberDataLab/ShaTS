import torch

from shats import ApproShaTS, BackgroundDatasetStrategy, FastShaTSIG


def _model_wrapper(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        x = x.unsqueeze(0)
    score = x.mean(dim=(1, 2))
    prob = torch.sigmoid(score)
    return torch.stack([1.0 - prob, prob], dim=1)


def _grad_model_wrapper(x: torch.Tensor) -> torch.Tensor:
    score = x.mean(dim=(1, 2))
    prob = torch.sigmoid(score)
    return torch.stack([1.0 - prob, prob], dim=1)


def _make_dataset():
    return [
        torch.tensor([[0.0, 0.1], [1.0, 0.2]]),
        torch.tensor([[1.0, 0.2], [0.0, 0.3]]),
        torch.tensor([[0.0, 0.4], [1.0, 0.5]]),
        torch.tensor([[1.0, 0.6], [0.0, 0.7]]),
    ]


def test_shats_accepts_explicit_background_dataset():
    background_dataset = _make_dataset()[:2]

    explainer = ApproShaTS(
        model_wrapper=_model_wrapper,
        background_dataset=background_dataset,
        grouping_strategy="feature",
        m=2,
        batch_size=2,
        device="cpu",
    )

    assert len(explainer.background_dataset) == 2
    assert explainer.groups_num == 2


def test_shats_can_infer_background_dataset_from_train_dataset():
    train_dataset = _make_dataset()

    explainer = ApproShaTS(
        model_wrapper=_model_wrapper,
        train_dataset=train_dataset,
        background_dataset_strategy=BackgroundDatasetStrategy.RANDOM,
        background_size=2,
        grouping_strategy="feature",
        m=2,
        batch_size=2,
        device="cpu",
        random_state=4,
    )

    assert len(explainer.background_dataset) == 2
    assert explainer.groups_num == 2


def test_fast_shats_ig_detects_binary_features_from_background_dataset():
    background_dataset = _make_dataset()

    explainer = FastShaTSIG(
        model_wrapper=_model_wrapper,
        grad_model_wrapper=_grad_model_wrapper,
        background_dataset=background_dataset,
        grouping_strategy="feature",
        m=2,
        batch_size=2,
        device="cpu",
        ig_steps=8,
        ig_class_idx=1,
    )

    assert explainer.categorical_feature_indices == [0]
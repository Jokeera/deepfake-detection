from models.dual_path import DualPathModel
from models.spatial_only import SpatialOnlyModel
from models.sequential import SequentialModel
from models.temporal_only import TemporalOnlyModel


_MODEL_REGISTRY = {
    "full": DualPathModel,
    "spatial": SpatialOnlyModel,
    "temporal": TemporalOnlyModel,
    "sequential": SequentialModel,
}


def build_model(cfg):
    """
    Фабрика моделей.

    Args:
        cfg: Config с параметрами модели.

    Returns:
        torch.nn.Module
    """
    if cfg.model_type not in _MODEL_REGISTRY:
        available = ", ".join(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Неизвестный тип модели: {cfg.model_type}. "
            f"Доступные варианты: {available}"
        )

    return _MODEL_REGISTRY[cfg.model_type](cfg)
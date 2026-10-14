"""vlnce_baselines: LaViRA zero-shot agent, mapping and multi-agent runner.

Submodules are imported lazily so that ``import vlnce_baselines`` stays cheap
(no torch / cv2 / GroundingDINO at package import). The historical package
attributes ``ZS_Evaluator_mp``, ``environments`` and ``Policy`` remain
available on first access.
"""
import importlib
import warnings

import matplotlib

# Previously applied as an import side effect of ZS_Evaluator_mp (imported
# eagerly here); kept so headless runs and log output behave as before.
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

_LAZY_SUBMODULES = {
    "ZS_Evaluator_mp": "vlnce_baselines.ZS_Evaluator_mp",
    "environments": "vlnce_baselines.common.environments",
    "Policy": "vlnce_baselines.models.Policy",
}

__all__ = list(_LAZY_SUBMODULES)


def __getattr__(name):
    target = _LAZY_SUBMODULES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(target)
    globals()[name] = module
    return module


def __dir__():
    return sorted(list(globals()) + __all__)

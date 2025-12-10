# trigger each task's @register_task
from . import binder as _binder  # noqa: F401
from . import monomer as _monomer  # noqa: F401
from .registry import get_task_class

__all__ = ["get_task_class"]

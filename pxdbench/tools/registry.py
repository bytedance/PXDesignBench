import os
from importlib import import_module
from typing import Callable, Optional, cast

from pxdbench.tools.ptx.interface import ProtenixAPI

PtxFactory = Callable[..., ProtenixAPI]

_FACTORIES: dict[str, PtxFactory] = {}


def register(name: str, factory: PtxFactory) -> None:
    _FACTORIES[name] = factory


def _load_from_dotted(mod_path: str, cls_name: str) -> PtxFactory:
    module = import_module(mod_path)
    cls = getattr(module, cls_name, None)
    if not callable(cls):
        raise TypeError(f"{cls_name} not found or not callable in {mod_path}")
    return cast(PtxFactory, cls)


def _parse_backend_spec(spec: str) -> PtxFactory:
    """
    Support:
      - 'pkg.mod:Class'
      - 'pkg.mod.Class'
    """
    if ":" in spec:
        left, cls_name = spec.split(":", 1)
        return _load_from_dotted(left, cls_name)

    if "." in spec:
        mod_path, cls_name = spec.rsplit(".", 1)
        return _load_from_dotted(mod_path, cls_name)

    raise ValueError(
        f"Unknown backend '{spec}'. "
        f"Expected a registered name, 'pkg.mod:Class' or 'pkg.mod.Class'."
    )


def get_backend(name: Optional[str] = None) -> ProtenixAPI:
    """
    Priority:
      1) explicit name
      2) env variable PXDBENCH_BACKEND
      3) default 'public'
    """
    chosen = name or os.getenv("PXDBENCH_BACKEND") or "public"

    if chosen in _FACTORIES:
        return _FACTORIES[chosen]

    return _parse_backend_spec(chosen)

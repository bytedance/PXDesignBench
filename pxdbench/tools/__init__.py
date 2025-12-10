from .ptx.ptx import ProtenixFilter
from .registry import register

register("public", ProtenixFilter)

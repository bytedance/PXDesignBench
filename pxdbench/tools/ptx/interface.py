from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ProtenixAPI(Protocol):
    def predict(self, input_json_path: str, **kw: Any): ...

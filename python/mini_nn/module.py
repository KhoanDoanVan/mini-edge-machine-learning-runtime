"""Base abstraction shared by all model components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from mini_ort import Tensor
from mini_ort.model import LayerSpec


class Module(ABC):

    def forward(
            self,
            input_tensor: Tensor
    ) -> Tensor:

        from mini_ort import InferenceSession

        with InferenceSession(self) as session:
            return session.run(input_tensor)

    def __call__(
            self,
            input_tensor: Tensor
    ) -> Tensor:
        return self.forward(input_tensor)


    @abstractmethod
    def layer_specs(self) -> tuple[LayerSpec, ...]:
        raise NotImplementedError

    def named_children(self) -> Iterator[
        tuple[
            str,
            Module
        ]
    ]:
        return iter(())

    def named_parameters(
            self,
            prefix: str = ""
    ) -> Iterator[
        tuple[
            str,
            Module
        ]
    ]:
        for child_name, child in self.named_children():
            yield from child.named_parameters(f"{prefix}{child_name}.")

    def state_dict(self) -> dict[str, Tensor]:
        return dict(self.named_parameters())

    def save(
            self,
            path: str | Path
    ):
        
        from mini_ort.serialization import save_model

        return save_model(
            self,
            path
        )


    def parameter_count(self) -> int:
        return sum(
            parameter.size for _, parameter in self.named_parameters()
        )

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        details = self.extra_repr()
        return f"{type(self).__name__}({details})" if details else type(self).__name__

    def summary(self) -> str:
        lines = [repr(self)]

        def append_chilren(
                module: Module,
                prefix: str
        ) -> None:
            children = tuple(module.named_children())

            for index, (name, child) in enumerate(children):
                is_last = index == len(children) - 1
                connector = "└──" if is_last else "├──"
                lines.append(f"{prefix}{connector} ({name}) {child!r}")
                child_prefix = f"{prefix}{'    ' if is_last else '│   '}"
                append_chilren(
                    child,
                    child_prefix
                )

            append_chilren(self, "")
            lines.append(f"Parameters: {self.parameter_count():,}")
            return "\n".join(lines)


    def save(
            self,
            path: str | Path
    ) -> Path:
        from mini_ort.serialization import save_model

        return save_model(self, path)
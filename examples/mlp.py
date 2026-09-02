"""Build and execute a tiny personalized MLP with the core API."""

from pathlib import Path

from mini_nn import Linear, ReLU, Sequential
from mini_ort import InferenceSession, Tensor


def build_model() -> Sequential:
    return Sequential(
        Linear(
            3,
            4,
            weight=Tensor.from_list(
                (3, 4),
                [1, 0, -1, 0.5, 0, 1, 1, -0.5, 1, 1, 0, 1],
            ),
            bias=Tensor.from_list((4,), [0, -1, 0.5, 0]),
        ),
        ReLU(),
        Linear(
            4,
            2,
            weight=Tensor.from_list(
                (4, 2),
                [1, -1, 0.5, 1, -1, 0, 0, 2],
            ),
            bias=Tensor.from_list((2,), [0.25, -0.5]),
        ),
    )



def main() -> None:
    model = build_model()
    model_path = model.save(Path("build/models/tiny_mlp.mer"))
    input_tensor = Tensor.from_list((1, 3), [1, 2, 3])
    batch = Tensor.from_list((2, 3), [1, 2, 3, 0, 1, -1])

    print(model.summary())
    with InferenceSession(model_path) as session:
        output = session.run(input_tensor)
        _ = session.run(input_tensor)  # Reuses the native output allocation.
        batch_output = session.run(batch)
        print(f"backend={session.backend_name}")
        print(f"input_shape={session.input_shape}")
        print(f"output_shape={session.output_shape}")
        print(f"model={model_path}")
    print(f"parameters={list(model.state_dict())}")
    print(f"output={output.to_list()}")
    print(f"batch_output={batch_output.to_list()}")


if __name__ == "__main__":
    main()

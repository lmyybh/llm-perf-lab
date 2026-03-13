import torch


def show_case(name: str, values: list[float]) -> None:
    tensor = torch.tensor(values, dtype=torch.float)
    print(f"--- {name} ---")
    print("input values:", values)
    print("tensor:", tensor)
    print("tensor == 1:", tensor == 1)
    print("torch.all(tensor == 1):", torch.all(tensor == 1))
    print()


show_case("all_true_after_rounding", [1.0, 1.0 + 1e-8, 1.0 + 2e-8, 1.0 + 5e-8])
show_case("mixed_true_false", [1.0, 1.0 + 1e-8, 1.0 + 1e-7, 1.0 + 1e-6])

import torch.nn as nn
from prettytable import PrettyTable

from models.model import FireNetMicroV2


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])

    print("\n--- Model Parameter Breakdown ---")
    for name, module in model.named_children():
        if not isinstance(module, nn.Module):
            continue
        params = sum(p.numel() for p in module.parameters())
        table.add_row([name, f"{params:,}"])

    print(table)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def main():
    model = FireNetMicroV2(num_classes=3)
    total, trainable = count_parameters(model)

    print("\nParameter count completed:")
    print("==========================================")
    print(f"Total Parameters:     {total:,}  ({total / 1e6:.2f} M)")
    print(f"Trainable Parameters: {trainable:,}  ({trainable / 1e6:.2f} M)")
    print("==========================================")


if __name__ == "__main__":
    main()

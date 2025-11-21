
from pyhessian import Hessian, HessianApproximator

from argparse import Namespace
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# TODO: put a toy model
def test_simple_approx():
    """A simple function to test code snippets."""
    torch.manual_seed(42)
    model = nn.Linear(10, 2, bias=True)
    model.eval()
    print(model)
    print(f"Model parameters: {[p.shape for p in model.parameters()]}")
    print(f"Total params: {sum(p.numel() for p in model.parameters())}")

    criterion = nn.CrossEntropyLoss()
    data = torch.randn(5, 10)
    target = torch.randint(0, 2, (5,))
    print(f"Targets: {target.shape} | Data: {data.shape}")

    dataset = TensorDataset(data, target)
    dataloader = DataLoader(dataset, batch_size=2)

    opts = Namespace(**{
        "ckpt": "init.pt",
        "device": "cpu",
    })
    hessian_comp = Hessian(
        model,
        criterion,
        dataloader=dataloader,
        cuda=(opts.device == "cuda")
    )

    exact = {"method": "exact", "log_every": 1}
    sample = {"method": "sampling", "step": 2, "log_every": 1}
    approximator = HessianApproximator(
        opts, hessian_comp, method_config=exact
    )
    hess_approx = approximator.run()
    print(f"Hessian matrix shape: {hess_approx.shape}")
    print(f"  range: [{hess_approx.min().item():.4e}, {hess_approx.max().item():.4e}]")
    # print(hess_approx)
    approximator.export_matrix()


if __name__ == "__main__":
    test_simple_approx()


from pyhessian import Hessian, HessianApproximator

import sys
from argparse import Namespace
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# TODO: put a toy model of which we know the hessian matrix
def test_simple_approx():
    """A simple function to test code snippets."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(6, 4, bias=True),
        nn.ReLU(),
        nn.Linear(4, 2, bias=True)
    )
    model.eval()
    print(model)
    # for p_name, p in model.named_parameters():
    #     if "bias" in p_name:
    #         p.requires_grad = False
    print(f"Model parameters: {[(p_name, p.shape, p.numel(), f"grad={p.requires_grad}") for p_name, p in model.named_parameters()]}")
    print(f"Total params: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    criterion = nn.CrossEntropyLoss()
    data = torch.randn(5, 6)
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

    # method = {"method": "exact", "log_every": 10}
    # method = {"method": "sampling", "step": 2, "log_every": 1}
    method = {"method": "block-wise", "log_every": 10}
    approximator = HessianApproximator(
        opts, hessian_comp, method_config=method
    )
    hess_approx_out = approximator.run()
    hess_approx = hess_approx_out['hess_approx']
    print(f"Hessian matrix shape: {hess_approx.shape}")
    print(f"  range: [{hess_approx.min().item():.4e}, {hess_approx.max().item():.4e}]")
    # # print(hess_approx)
    # approximator.export_matrix()


if __name__ == "__main__":
    try:
        test_simple_approx()
    except Exception:
        import ipdb, traceback
        traceback.print_exc()
        ipdb.post_mortem(sys.exc_info()[2])

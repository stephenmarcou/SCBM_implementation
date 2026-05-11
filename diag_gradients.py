#!/usr/bin/env python3
import sys
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from utils.intervention import SCBM_Strategy
from models.models import SCBM_residual

# Ensure repo root on path if running from scripts/
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

def make_config():
    model_ns = SimpleNamespace()
    model_ns.model = "scbm_residual"
    model_ns.encoder_arch = "FCNN"
    model_ns.head_arch = "linear"
    model_ns.training_mode = "joint"
    model_ns.concept_learning = "hard"
    model_ns.num_monte_carlo = 8
    model_ns.straight_through = True
    model_ns.j_epochs = 1
    model_ns.t_epochs = 1
    model_ns.cov_type = "amortized"
    model_ns.level = 0.9
    model_ns.model_directory = ""
    data_ns = SimpleNamespace()
    data_ns.num_concepts = 5
    data_ns.num_residuals = 2
    data_ns.num_classes = 3   # use multiclass so returned logits are log-probs compatible with nll_loss
    data_ns.num_covariates = 16
    cfg = SimpleNamespace(model=model_ns, data=data_ns)
    return cfg

def main(device_str="cpu"):
    device = torch.device(device_str)
    cfg = make_config()
    # instantiate model
    model = SCBM_residual(cfg).to(device).train()

    B = 2
    # dummy features
    x = torch.randn(B, cfg.data.num_covariates, device=device)
    # run forward to get extension-return values
    (
        c_res_mcmc_prob,
        c_res_mcmc,
        c_res_mcmc_logit,
        c_res_triang_cov,
        y_pred_logits,
        c_res_mu,
    ) = model.forward(x, epoch=0, return_L_int_extension=True)

    # build cov from triangular
    c_res_cov = torch.matmul(c_res_triang_cov, c_res_triang_cov.transpose(-1, -2))

    # dummy ground-truth concepts (no residuals)
    concepts_true = torch.randint(0, 2, (B, cfg.data.num_concepts), device=device).float()
    concepts_mask = torch.ones_like(concepts_true, device=device)  # intervene on all concepts

    # choose simple_percentile strategy to avoid CPU optimizer path
    interv = SCBM_Strategy("simple_perc", train_loader=None, model=None, device=device, config=cfg)

    # compute intervention (this mirrors compute_L_int_extension_loss)
    (
        c_res_interv_mu,
        c_res_interv_cov,
        c_res_mcmc_probs,
        c_res_mcmc_logits,
    ) = interv.compute_intervention(c_res_mu, c_res_cov, concepts_true, concepts_mask)

    # straight-through intervene via model method
    target_pred_logits = model.intervene_straight_through(c_res_mcmc_probs, c_res_mcmc_logits)

    # build dummy targets (multiclass)
    targets = torch.randint(0, cfg.data.num_classes, (B,), device=device, dtype=torch.long)

    # For multiclass intervene_straight_through returns log-probs (log of averaged softmax) -> use nll_loss
    loss = F.nll_loss(target_pred_logits, targets)

    # zero grads, backward, inspect
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()
    loss.backward()

    print("Parameter gradient summary (name, shape, grad norm):")
    for name, p in model.named_parameters():
        g = p.grad
        if g is None:
            print(f"- {name}: grad=None")
        else:
            gnorm = g.norm().item()
            print(f"- {name}: shape={tuple(p.shape)}, grad_norm={gnorm:.6e}")

    # Quick module-level summary
    def module_grad_norm(mod):
        s = 0.0
        any_grad = False
        for p in mod.parameters():
            if p.grad is not None:
                any_grad = True
                s += p.grad.norm().item() ** 2
        return (any_grad, s ** 0.5)

    for mod_name in ("encoder", "mu_concepts_residuals", "sigma_concepts_residuals", "head"):
        mod = getattr(model, mod_name, None)
        if mod is None:
            continue
        any_grad, norm = module_grad_norm(mod)
        print(f"Module `{mod_name}`: any_grad={any_grad}, total_grad_norm={norm:.6e}")

if __name__ == "__main__":
    # choose GPU if available
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    main(dev)
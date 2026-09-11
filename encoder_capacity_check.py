"""
Test 1: can the encoder read the hidden concept X off the pixels at all?

This is a capacity check, not a model. It trains the *same* encoder the Residual
SCBM uses, on the *same* input, under the *same* optimisation, with one linear
logit on top of the shared feature vector -- supervised directly on X. The only
thing that differs from the real run is the loss.

That "only the loss differs" is the whole point, so this script goes out of its
way to inherit rather than restate:

  input          the full 4-channel stack (d1, d2, d3, d4) in the dataset's own
                 channel order, distractors included. The distractors carry no
                 information about X and the encoder is free to ignore them, but
                 keeping them holds the input distribution and the first-layer
                 weight shape identical to the real model -- and puts the
                 frozen-encoder probe in Test 2 on exactly the same footing.
  corruption     whatever the dataset does. Noise is drawn deterministically per
                 (sample, channel) from `seed`, so it is fixed across epochs, and
                 this script gets that for free by using the same dataset class.
                 If that ever changes to fresh-noise-per-epoch, both change
                 together.
  splits         same generator seeding (`reset_random_seeds`) and the same
                 `get_MNIST_add_cov_datasets` call train.py makes, so the split
                 fingerprints printed at startup match the real run's.
  normalisation  the dataset's own (`/255.0`, nothing further).
  loaders        batch sizes, shuffle, drop_last, workers, pin_memory and
                 generator copied from utils/data.py's construction.
  optimisation   optimiser, learning rate, weight decay, StepLR schedule and
                 epoch budget read from the composed model config -- or from a
                 finished run's log.txt with --run, which is airtight.
  loop order     validate, check early stopping, train, step the scheduler --
                 train.py's ordering, including its validate-before-train quirk,
                 so epoch counts mean the same thing in both.

The one deliberate deviation: early stopping tracks val accuracy on X, since X
is the only target here. train.py tracks `y_accuracy` on the task label.

Read the result as a statement about *recoverability* -- what a supervised
read-out can extract from these pixels -- not about what an unsupervised
residual channel would organise itself around.

Usage
-----
    # match a finished run's hyperparameters exactly
    python encoder_capacity_check.py --run planted_xor_MNIST_ADD_carry_corr_strength_02_global_no_prec_emp_perc_residuals_1_L_int_extension_loss_weight_1_2026-09-07_16-41-10_ec22a

    # or compose from the configs and set the corruption level by hand
    python encoder_capacity_check.py --corruption-strength 0.5

    # a supervised concept instead, as a sanity check that training works
    python encoder_capacity_check.py --target A

    # secondary question: how much do the distractor channels cost the encoder?
    # (run this only if the matched 4-channel result comes out ambiguous)
    python encoder_capacity_check.py --num-covariates 2
"""

import argparse
import ast
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from datasets.MNIST_add_cov_dataset import get_MNIST_add_cov_datasets
from models.models import IntCEMMNISTEncoder
from utils.utils import reset_random_seeds

REPO_ROOT = Path(__file__).resolve().parent


def pick_device():
    """train.py's device selection."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_config(args):
    """
    Compose `+data=mnist_add +model=SCBM_RES`, then either overlay a finished
    run's recorded config (--run) or apply the handful of overrides this script
    exposes. Returns the full config; `.data` and `.model` are read from it the
    same way train.py reads them.
    """
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        config = compose(
            config_name="config", overrides=["+data=mnist_add", "+model=SCBM_RES"]
        )

    if args.run:
        run_dir = REPO_ROOT / "experiments" / "scbm_residual" / "MNIST-Add-Cov" / args.run
        log_txt = run_dir / "log.txt"
        if not log_txt.exists():
            raise SystemExit(f"No log.txt at {log_txt}")
        recorded = ast.literal_eval(log_txt.read_text().splitlines()[0])
        # Overlay wholesale: every data and model key the run actually used,
        # so nothing silently falls back to a config default that has since
        # been edited. data_path is the exception -- the run's is a cluster
        # path that does not exist here.
        data_path = config.data.data_path
        config.data = OmegaConf.merge(config.data, recorded["data"])
        config.model = OmegaConf.merge(config.model, recorded["model"])
        config.data.data_path = data_path
        config.seed = recorded.get("seed", config.seed)
        config.workers = recorded.get("workers", config.workers)
        print(f"hyperparameters from: {args.run}")

    # Applied after the overlay, deliberately: --run pins everything else, so
    # this varies corruption as a single clean axis against a matched baseline.
    if args.corruption_strength is not None:
        prob = 0.0 if args.corruption_strength == 0 else 1.0
        config.data.corruption_strength = args.corruption_strength
        config.data.test_corruption_strength = args.corruption_strength
        config.data.corruption_probability = prob
        config.data.test_corruption_probability = prob
    if args.num_covariates is not None:
        config.data.num_covariates = args.num_covariates

    if args.seed is not None:
        config.seed = args.seed
    if args.epochs is not None:
        config.model.j_epochs = args.epochs
    return config


def target_extractor(dataset, target):
    """
    Map a batch dict to a float target vector. X comes from the oracle metadata;
    a concept is looked up by short name rather than a hardcoded index, so this
    survives the re-indexing that `removed_concepts` causes.
    """
    if target == "X":
        return (lambda batch: batch["X"]), "X (the carry, oracle-only)"

    short_names = [n.split("::")[0] for n in dataset.concept_names()]
    if target not in short_names:
        raise SystemExit(
            f"--target {target!r} is neither X nor an exposed concept {short_names}."
        )
    idx = short_names.index(target)
    return (lambda batch: batch["concepts"][:, idx]), f"{target} (supervised concept)"


def concept_only_ceiling(dataset, extract, workers):
    """
    Best accuracy AND AUROC attainable from (A, B) alone, measured on this split.

    The optimal concept-only predictor scores each sample by its (A, B) cell's
    P(target = 1). For accuracy that collapses to the cell majority; averaging
    those gives 0.80 on `hidden_carry`, since X is pinned wherever A == B and the
    remaining half splits 60/40. For AUROC the same scores give ~0.879 -- four
    samples in a cell are tied, so the two ambiguous cells contribute half credit.

    Both bars matter. Accuracy is coarse here: the whole gap between a predictor
    that knows nothing and one that reads X perfectly is only 0.20 wide, and
    thresholding at 0.5 throws away everything the model knows about the
    ambiguous cells. AUROC uses the ranking and is the more sensitive test of
    whether any pixel information about X got through at all.
    """
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=workers)
    cell_ids, targets = [], []
    for batch in loader:
        ab = batch["concepts"][:, :2].long()
        cell_ids.append(ab[:, 0] * 2 + ab[:, 1])
        targets.append(extract(batch).long())

    cell_ids = torch.cat(cell_ids)
    targets = torch.cat(targets)

    # P(target = 1 | cell), the optimal concept-only score.
    scores = torch.zeros_like(targets, dtype=torch.float)
    correct = 0
    for key in range(4):
        mask = cell_ids == key
        n = int(mask.sum())
        if n == 0:
            continue
        pos = int(targets[mask].sum())
        scores[mask] = pos / n
        correct += max(pos, n - pos)

    acc = correct / len(targets)
    auroc = roc_auc_score(targets.numpy(), scores.numpy())
    return acc, auroc


@torch.no_grad()
def evaluate(model, loader, extract, device):
    model.eval()
    logits, targets = [], []
    for batch in loader:
        logits.append(model(batch["features"].to(device)).squeeze(1).float().cpu())
        targets.append(extract(batch).float())

    logits = torch.cat(logits)
    targets = torch.cat(targets)
    acc = ((logits > 0).float() == targets).float().mean().item()
    auroc = roc_auc_score(targets.numpy(), logits.numpy())
    return acc, auroc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None,
                        help="run folder under experiments/scbm_residual/MNIST-Add-Cov/ "
                             "to copy data and model hyperparameters from")
    parser.add_argument("--target", default="X",
                        help="X (default) or an exposed concept: A, B, D1, D2")
    parser.add_argument("--corruption-strength", type=float, default=None,
                        help="all digit channels, train and test; composes with --run")
    parser.add_argument("--num-covariates", type=int, default=None,
                        help="secondary experiment: 2 drops the distractor channels; "
                             "composes with --run")
    parser.add_argument("--epochs", type=int, default=None,
                        help="overrides model.j_epochs")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = build_config(args)
    # train.py seeds once, up front, and hands the generator to the loaders.
    gen = reset_random_seeds(config.seed)
    # The conv backbone has no deterministic CUDA kernel for its backward pass;
    # reset_random_seeds turns on strict determinism, so relax that one bit.
    torch.use_deterministic_algorithms(False)
    device = pick_device()

    trainset, valset, testset = get_MNIST_add_cov_datasets(
        config.data, config.incomplete, seed=config.seed
    )
    extract, target_label = target_extractor(trainset, args.target)

    m = config.model
    train_loader = DataLoader(
        trainset, batch_size=m.train_batch_size, shuffle=True,
        num_workers=config.workers, pin_memory=True, generator=gen,
        drop_last=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        valset, batch_size=m.val_batch_size, shuffle=True,
        num_workers=config.workers, pin_memory=True, generator=gen,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        testset, batch_size=m.val_batch_size, num_workers=config.workers, generator=gen,
    )

    # The encoder up to the shared feature vector, plus one linear logit. Not
    # the concept head, not the residual head, no extra hidden layer -- so
    # "encoder capacity" means the same thing here as in the real model.
    in_channels = int(config.data.num_covariates)
    encoder = IntCEMMNISTEncoder(in_channels=in_channels, output_dim=128)
    model = nn.Sequential(encoder, nn.Linear(128, 1)).to(device)

    optimizer = optim.Adam([{
        "params": filter(lambda p: p.requires_grad, model.parameters()),
        "lr": m.learning_rate,
        "weight_decay": m.weight_decay,
    }])
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=m.decrease_every, gamma=1 / m.lr_divisor,
    )
    criterion = nn.BCEWithLogitsLoss()

    ceiling_acc, ceiling_auroc = concept_only_ceiling(trainset, extract, config.workers)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"experiment      : {config.data.experiment}")
    print(f"target          : {target_label}")
    print(f"input           : [{in_channels}, 28, 28]  (channels d1..d{in_channels})")
    print(f"corruption      : {config.data.corruption} @ "
          f"{config.data.corruption_strength}, p={config.data.corruption_probability}, "
          f"channels {list(trainset.corruption_channels)}")
    print(f"encoder         : IntCEMMNISTEncoder + Linear(128, 1) "
          f"({n_params:,} params)")
    print(f"optimiser       : {m.optimizer} lr={m.learning_rate} "
          f"wd={m.weight_decay}, StepLR(every {m.decrease_every}, /{m.lr_divisor})")
    print(f"batch size      : train {m.train_batch_size} (drop_last), "
          f"val/test {m.val_batch_size}")
    print(f"epochs          : {m.j_epochs}, early stop patience "
          f"{m.early_stopping_patience} on val {args.target} accuracy")
    print(f"seed / device   : {config.seed} / {device}")
    print(f"concept-only ceiling (best from A,B alone): "
          f"accuracy {ceiling_acc:.4f}, AUROC {ceiling_auroc:.4f}\n")
    print(f"{'epoch':>6}  {'train_loss':>10}  {'val_acc':>8}  {'val_auroc':>9}  {'lr':>8}")

    best = {"val_acc": float("-inf"), "epoch": -1, "state": None}
    epochs_without_improvement = 0

    # train.py's ordering: validate, decide, train, step. The validation at
    # epoch 0 therefore scores the untrained model, and the checkpoint kept is
    # always the state at the end of the previous epoch.
    for epoch in range(0, m.j_epochs):
        val_acc, val_auroc = evaluate(model, val_loader, extract, device)

        if val_acc > best["val_acc"]:
            best = {
                "val_acc": val_acc,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()},
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= m.early_stopping_patience:
            print(f"Early stopping: no improvement for "
                  f"{m.early_stopping_patience} epochs. "
                  f"Best val accuracy {best['val_acc']:.4f}")
            break

        model.train()
        running, seen = 0.0, 0
        for batch in train_loader:
            x = batch["features"].to(device)
            t = extract(batch).float().to(device)

            optimizer.zero_grad()
            loss = criterion(model(x).squeeze(1), t)
            loss.backward()
            optimizer.step()

            running += loss.item() * x.size(0)
            seen += x.size(0)

        lr_scheduler.step()
        print(f"{epoch:>6}  {running / seen:>10.4f}  {val_acc:>8.4f}  "
              f"{val_auroc:>9.4f}  {lr_scheduler.get_last_lr()[0]:>8.2e}")

    model.load_state_dict(best["state"])
    test_acc, test_auroc = evaluate(model, test_loader, extract, device)

    print(f"\nbest epoch      : {best['epoch']} (val accuracy {best['val_acc']:.4f})")
    print(f"test accuracy   : {test_acc:.4f}")
    print(f"test AUROC      : {test_auroc:.4f}")

    if args.target == "X":
        acc_margin = test_acc - ceiling_acc
        auroc_margin = test_auroc - ceiling_auroc
        print(f"\n{'':16}{'model':>10}  {'ceiling':>10}  {'margin':>10}")
        print(f"{'accuracy':16}{test_acc:>10.4f}  {ceiling_acc:>10.4f}  "
              f"{acc_margin:>+10.4f}")
        print(f"{'AUROC':16}{test_auroc:>10.4f}  {ceiling_auroc:>10.4f}  "
              f"{auroc_margin:>+10.4f}")

        # AUROC is the sensitive test; accuracy corroborates. Thresholds are
        # judgement calls, so the margins are printed above either way -- read
        # them, do not just read the verdict.
        if auroc_margin > 0.03 and acc_margin > 0.01:
            verdict = ("the encoder recovers X from the pixels beyond what (A,B) "
                       "already imply. A flat residual cross-block would be a real "
                       "negative about the architecture.")
        elif auroc_margin > 0.01:
            verdict = ("MARGINAL. The encoder gets some pixel information about X "
                       "through, but barely. A weak residual result here is not "
                       "clearly attributable to the residual channel.")
        else:
            verdict = ("the encoder does NOT clear the concept-only ceiling, so a "
                       "flat residual says nothing about the residual channel. Fix "
                       "this before reading anything off the cross-block.")
        print(f"\nverdict: {verdict}")


if __name__ == "__main__":
    main()

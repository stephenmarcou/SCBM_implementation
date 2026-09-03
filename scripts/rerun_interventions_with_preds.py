"""Re-run the intervention sweep of already-trained runs, dumping per-sample predictions.

Why this exists: intervention_log.txt records only the average over the evaluation split, and
on Waterbirds that average is precisely what hides the question - a head reading the background
and a head reading the bird can land on the same number. inference.save_intervention_predictions
writes the per-sample predictions behind every point of the curve so it can be re-read within
the four (bird, background) cells afterwards. Nothing is retrained; each run costs one forward
pass over the split plus the (cached) sweep steps.

The overrides that matter are recovered from each run's own log.txt rather than typed out, since
getting one wrong is silent-ish and expensive: inference.py restores only pkl_file_dir,
num_concepts and num_residuals, so data.binary_target, model.concept_learning and
model.inter_strategy have to be passed back or the model is rebuilt with the wrong head / the
curve is measured with a different intervention strategy than the one already plotted.

    python scripts/rerun_interventions_with_preds.py --dry-run          # print the commands
    python scripts/rerun_interventions_with_preds.py                    # all Waterbirds runs
    python scripts/rerun_interventions_with_preds.py --runs cbm/Mateo_025_binary_target_...
    python scripts/rerun_interventions_with_preds.py --inter-strategy from-log

Every run is swept with emp_perc by default, rather than with the strategy its log.txt records.
Two reasons. Curves measured with different strategies cannot be read against each other, so a
sweep meant to be compared across model families has to pin one, and emp_perc is the one every
family supports. And the strategy in the config is not necessarily the strategy the run's curve
was measured with: a run swept post-hoc with an explicit model.inter_strategy override keeps the
training-time value in log.txt, so trusting the config silently re-measures the curve with a
different strategy than the one already plotted - and since inference.py opens
intervention_log.txt with "w", it overwrites the original in the process.

--inter-strategy takes a strategy name, or one of two sentinels:
    from-log      the strategy the run's existing intervention_log.txt header records, i.e.
                  reproduce the curve that is in the folder (falls back to the config when the
                  run has no log yet)
    from-config   the strategy in log.txt, i.e. what the model was trained with

emp_perc is also the fix for the soft CBMs, whose recorded strategy is the literal string
"simple_perc,emp_perc" - define_strategy matches strategy names exactly, so that value falls
through every branch and is used as if it were a strategy object (AttributeError: 'str' object
has no attribute 'compute_intervention_cbm'). Their existing intervention_log.txt files stop
after the zero-intervention point for exactly this reason.

Hard CBMs are the exception no strategy choice can reach: define_strategy pins
concept_learning=hard to HardCBMStrategy before it ever looks at the strategy name, so those runs
stay on the hard strategy and their dump lands in hard_random/ rather than emp_perc_random/.

Since inference.py opens intervention_log.txt with "w", a re-run under a different strategy would
drop the curve already in the folder; the existing log is copied to
intervention_log_<its own strategy>.txt first (once - an existing copy is never overwritten).
"""

import argparse
import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Swept with this unless --inter-strategy says otherwise: the one strategy every model family
# supports, so the curves stay comparable across them.
DEFAULT_STRATEGY = "emp_perc"
# Sentinels accepted by --inter-strategy in place of a strategy name.
STRATEGY_FROM_LOG = "from-log"
STRATEGY_FROM_CONFIG = "from-config"

# model.model as recorded in log.txt -> the configs/model/<name>.yaml to compose with.
MODEL_CONFIGS = {
    "cbm": "CBM",
    "cbm_residual": "CBM_RES",
    "scbm": "SCBM",
    "scbm_residual": "SCBM_RES",
    "cem": "CEM",
    "ar": "AR",
}


def read_run_config(run_path):
    """The full config of a finished run: the first line of its log.txt."""
    with open(run_path / "log.txt") as f:
        return ast.literal_eval(f.readline().strip())


def logged_strategy(log_file):
    """The strategy an existing intervention_log.txt was measured with, from its header.

    Read from the log rather than from the run's config because they disagree: several runs were
    swept post-hoc with an explicit model.inter_strategy override, so the config records what the
    model was trained with and the header records what the curve in that file actually is.
    """
    with open(log_file) as f:
        for line in f:
            m = re.search(r"Intervention strategy: \['([^']*)'\]", line)
            if m:
                return m.group(1)
    return None


def backup_existing_log(run_path):
    """Keep the curve already in the folder before inference.py truncates it."""
    log_file = run_path / "intervention_log.txt"
    if not log_file.exists():
        return None
    strategy = logged_strategy(log_file) or "unknown"
    backup = run_path / f"intervention_log_{strategy.replace(',', '_')}.txt"
    if backup.exists():
        return None
    shutil.copy2(log_file, backup)
    return backup


def resolve_strategy(run_path, cfg, requested):
    """The strategy to sweep with: a name, or one of the two sentinels resolved against the run."""
    if requested == STRATEGY_FROM_CONFIG:
        return cfg["model"]["inter_strategy"]
    if requested == STRATEGY_FROM_LOG:
        # What the curve in the folder was actually measured with, which is not always what the
        # model was trained with.
        return (logged_strategy(run_path / "intervention_log.txt")
                or cfg["model"]["inter_strategy"])
    return requested


def build_command(run_path, dataset, inter_strategy=DEFAULT_STRATEGY, python=sys.executable):
    """Hydra command re-running one run's interventions with the prediction dump on."""
    cfg = read_run_config(run_path)
    model_cfg = MODEL_CONFIGS.get(cfg["model"]["model"])
    if model_cfg is None:
        raise ValueError(f"{run_path.name}: no configs/model entry for {cfg['model']['model']}")

    cmd = [
        python, "inference.py",
        f"+model={model_cfg}",
        f"+data={dataset}",
        f"inference.ex_name={run_path.name}",
        "run_inference=False",
        "run_interventions=True",
        "inference.save_intervention_predictions=True",
    ]
    if cfg.get("incomplete"):
        # Without this the pkl_file_dir / num_concepts / num_residuals of the run are not
        # recovered from log.txt and the checkpoint will not load.
        cmd.append("incomplete=True")
    if cfg["data"].get("binary_target"):
        cmd.append("data.binary_target=True")
    if "concept_learning" in cfg["model"]:
        cmd.append(f"model.concept_learning={cfg['model']['concept_learning']}")
    # Quoted: several runs record a comma-separated strategy, which Hydra would otherwise read
    # as a multirun sweep and refuse.
    strategy = resolve_strategy(run_path, cfg, inter_strategy)
    cmd.append(f"model.inter_strategy='{strategy}'")
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Waterbirds")
    parser.add_argument(
        "--experiment-dir", default="experiments",
        help="Root holding <model_type>/<dataset>/<run>/ (default: experiments).",
    )
    parser.add_argument(
        "--runs", nargs="*", default=None,
        help="Runs to re-run as <model_type>/<run_dir>. Default: every run of the dataset "
             "that has a model.pth.",
    )
    parser.add_argument(
        "--inter-strategy", default=DEFAULT_STRATEGY,
        help=f"Intervention strategy to sweep every run with (default: {DEFAULT_STRATEGY}, the "
             f"one every model family supports). '{STRATEGY_FROM_LOG}' reproduces the strategy "
             f"each run's existing intervention_log.txt was measured with, "
             f"'{STRATEGY_FROM_CONFIG}' uses the one it was trained with. Hard CBMs ignore all "
             "of these - see the module docstring.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the commands and exit.")
    args = parser.parse_args()

    root = Path(args.experiment_dir)
    if args.runs:
        run_paths = [root / r.split("/")[0] / args.dataset / r.split("/", 1)[1] for r in args.runs]
    else:
        run_paths = sorted(
            p.parent for p in root.glob(f"*/{args.dataset}/*/model.pth")
        )
    if not run_paths:
        raise SystemExit(f"No runs found under {root}/*/{args.dataset}/.")

    failures = []
    for i, run_path in enumerate(run_paths, 1):
        if not (run_path / "log.txt").exists():
            failures.append((run_path.name, "no log.txt"))
            continue
        cmd = build_command(run_path, args.dataset, inter_strategy=args.inter_strategy)
        print(f"\n[{i}/{len(run_paths)}] {run_path}", flush=True)
        print("  " + " ".join(cmd), flush=True)
        if args.dry_run:
            continue
        backup = backup_existing_log(run_path)
        if backup is not None:
            print(f"  kept the existing curve as {backup.name}", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            failures.append((run_path.name, f"exit {result.returncode}"))

    if failures:
        print(f"\n{len(failures)} run(s) failed:")
        for name, why in failures:
            print(f"  {name}: {why}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

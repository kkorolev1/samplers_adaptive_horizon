import os
import argparse
import sys


def create_argument_combinations():
    """Create all combinations of arguments for grid search."""

    # Define the parameter grid
    param_grid = []
    fixed_params = {
        "algorithm": "gfn_non_acyclic",
        "target": "gaussian_mixture9",
        "algorithm.weight_decay": 1e-8,
        "algorithm.reg_coef": 1e-4,
        "algorithm.no_term": False,
        "algorithm.model.learn_fwd_corrections": True,
        "algorithm.model.shared_model": False,
        "algorithm.num_steps": 200,
        "algorithm.eval_max_steps": 200,
        "algorithm.local_search.use": True,
        "algorithm.step_size": "1e-3",
        "algorithm.bwd_step_size": "1e-3",
        "algorithm.model.outer_clip": "100",
        "algorithm.model.gamma": "0.5",
        "algorithm.buffer.bwd_to_fwd_ratio": 2,
    } 

    # Run with fixed params only
    if len(param_grid) == 0:
        param_grid = [{}]

    combinations = []
    
    api_key = os.getenv("COMETML_API_KEY")
    assert api_key is not None, "Comet API key is not set"
    workspace = os.getenv("COMETML_WORKSPACE")
    assert workspace is not None, "Comet workspace is not set"

    for params in param_grid:
        cmd = [
            f"COMETML_API_KEY='{api_key}' COMETML_WORKSPACE='{workspace}'",
            "python",
            "run.py",
        ]

        # Add fixed parameters
        for name, value in fixed_params.items():
            cmd.append(f"{name}={str(value)}")

        # # Add grid search parameters
        for name, value in params.items():
            cmd.append(f"{name}={str(value)}")

        # Add boolean flags (we'll create separate combinations for these)
        combinations.append(cmd)

    return [" ".join(comb) for comb in combinations]


def main(args):
    python_commands = create_argument_combinations()
    for python_command in python_commands:
        version = 510
        constraint = args.node
        name = "samplers_adaptive_horizon"

        python_prefix = ""
        if args.gpus == 0:
            python_prefix = "JAX_PLATFORMS=cpu"
        python_command = f"{python_prefix} {python_command}"
        sbatch_command = (
            f"sbatch -A proj_1650 -c {args.cpu_cores} -G {args.gpus} --job-name={name}"
            f"--error=sbatch_logs_{version}/{name}/%j.err --output=sbatch_logs_{version}/{name}/%j.log "
            f'--constraint="{constraint}" --time=0-1:00:00'
        )
        if constraint in ["type_g", "type_h"]:
            sbatch_command += " --reservation rocky"
        print(python_command, "\n\n")
        if not args.dry_run:
            os.system(f'{sbatch_command} --wrap="{python_command}"')


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--cpu_cores", type=int, default=12)
    p.add_argument("--gpus", type=int, default=0)
    p.add_argument("--node", type=str, default="type_g")
    args = p.parse_args()
    main(args)

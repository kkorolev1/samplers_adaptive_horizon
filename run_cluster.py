import os
import argparse


def create_argument_combinations():
    """Create all combinations of arguments for grid search."""

    # Define the parameter grid
    param_grid = [{"algorithm.reg_coef": 0.0}]
    fixed_params = {
        "algorithm": "gfn_non_acyclic",
        "target": "gaussian_mixture9",
        "algorithm.weight_decay": 1e-0,
        "algorithm.no_term": False,
        "algorithm.batch_size": 256
    } 

    # Run with fixed params only
    if len(param_grid) == 0:
        param_grid = [{}]

    combinations = []
    for params in param_grid:
        cmd = [
            f"COMETML_API_KEY='rsWnsJtGaRo3twpTYJlHagKLm' COMETML_WORKSPACE='kirill-korolev'",
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

        sbatch_command = (
            f"sbatch -A proj_1650 -c {args.cpu_cores} -G {args.gpus} --job-name={name}"
            f"--error=sbatch_logs_{version}/{name}/%j.err --output=sbatch_logs_{version}/{name}/%j.log "
            f'--constraint="{constraint}" --time=0-1:00:00 --reservation rocky'
        )
        print(python_command, "\n\n")
        if not args.dry_run:
            os.system(f'{sbatch_command} --wrap="{python_command}"')


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--cpu_cores", type=int, default=12)
    p.add_argument("--gpus", type=int, default=0)
    p.add_argument("--node", type=str, default="type_g", choices=["type_g", "type_h"])
    args = p.parse_args()
    main(args)

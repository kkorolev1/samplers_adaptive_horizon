def print_results(step, logger, config):
    if config.verbose:
        string = f"Step {int(step)}: ELBO {float(logger['KL/elbo'][-1]):.4f}; "
        if "KL/eubo" in logger and len(logger["KL/eubo"]) > 0:
            string += f"EUBO {float(logger['KL/eubo'][-1]):.4f}; "
        if "discrepancies/sd" in logger and len(logger["discrepancies/sd"]) > 0:
            string += f"SD {float(logger['discrepancies/sd'][-1]):.4f}; "
        if (
            "max_traj_length/reverse" in logger
            and len(logger["max_traj_length/reverse"]) > 0
        ):
            string += f"MaxLen {float(logger['max_traj_length/reverse'][-1]):.4f}; "
        if (
            "mean_traj_length/reverse" in logger
            and len(logger["mean_traj_length/reverse"]) > 0
        ):
            string += f"MeanLen {float(logger['mean_traj_length/reverse'][-1]):.4f}; "
        print(string)

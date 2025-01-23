#experiments.py

import pandas as pd
from tabulate import tabulate
from main_deep_rl_experiments import run_experiment

def run_batch_experiments(num_runs=5):
    """
    Runs DHR+RL experiment multiple times (no plots), collecting performance stats,
    and prints them in a rounded table.
    """
    all_results = []
    for i in range(1, num_runs + 1):
        print(f"\n--- Experiment {i}/{num_runs} ---")
        res = run_experiment(no_plot=True)
        res["Trial"] = i
        all_results.append(res)

    df = pd.DataFrame(all_results)

    # We want training and testing for RL & EW, so let's define column order
    col_order = [
        "Trial",
        "Train_RL_AnnRet", "Train_RL_Vol", "Train_RL_Sharpe", "Train_RL_MaxDD",
        "Train_EW_AnnRet", "Train_EW_Vol", "Train_EW_Sharpe", "Train_EW_MaxDD",
        "Test_RL_AnnRet",  "Test_RL_Vol", "Test_RL_Sharpe", "Test_RL_MaxDD",
        "Test_EW_AnnRet", "Test_EW_Vol", "Test_EW_Sharpe", "Test_EW_MaxDD"
    ]

    # Filter to only existing columns in case something is missing
    cols_exist = [c for c in col_order if c in df.columns]
    df = df[cols_exist]

    # Round to 3 decimal places
    df = df.round(3)

    print("\n=== BATCH RESULTS ===")
    print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))

def main():
    run_batch_experiments(num_runs=5)

if __name__ == "__main__":
    main()

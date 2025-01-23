#experiments_returns_only.py

import pandas as pd
from tabulate import tabulate
from main_deep_rl_experiments import run_experiment

def run_batch_experiments(num_runs=5):
    """
    Runs DHR+RL experiment multiple times (no plots), collecting performance stats,
    prints them in a table, and shows success rate for RL beating EW on test returns.
    """
    all_results = []
    for i in range(1, num_runs + 1):
        print(f"\n--- Experiment {i}/{num_runs} ---")
        res = run_experiment(no_plot=True)
        res["Trial"] = i
        all_results.append(res)

    df = pd.DataFrame(all_results)

    # Reorder columns if desired
    col_order = [
        "Trial",
        "Train_RL_AnnRet", "Train_EW_AnnRet",
        "Test_RL_AnnRet",  "Test_EW_AnnRet"
    ]
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    # Round to 3 decimals
    df = df.round(3)

    # Calculate RL vs. EW success rate (Test)
    # Mark 1 if RL annual return is greater than EW, else 0
    if "Test_RL_AnnRet" in df.columns and "Test_EW_AnnRet" in df.columns:
        df["TestSuccess"] = (df["Test_RL_AnnRet"] > df["Test_EW_AnnRet"]).astype(int)
        success_rate = df["TestSuccess"].mean() * 100
    else:
        success_rate = None

    print("\n=== BATCH RESULTS ===")
    print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))

    # Print success rate if we have both columns
    if success_rate is not None:
        print(f"\nSuccess rate (RL beats EW in Test): {success_rate:.1f}%")

def main():
    run_batch_experiments(num_runs=5)

if __name__ == "__main__":
    main()

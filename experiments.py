import pandas as pd
from tabulate import tabulate  # pip install tabulate if not installed
from main_deep_rl_experiments import run_experiment

def run_batch_experiments(num_runs=5):
    """
    Runs DHR+RL experiment multiple times (no plots), collecting performance stats.
    Prints final results in a tabular format.
    """
    all_results = []
    for i in range(1, num_runs + 1):
        print(f"\n--- Experiment {i}/{num_runs} ---")
        # Call run_experiment with no_plot=True to skip plotting
        res = run_experiment(no_plot=True)
        # Add trial number to results
        res['Trial'] = i
        all_results.append(res)

    df = pd.DataFrame(all_results)

    # If you have columns like RL_AnnualizedReturn, RL_Volatility, RL_Sharpe, RL_MaxDD
    # you can order them here
    col_order = [
        "Trial",
        "RL_AnnualizedReturn",
        "RL_Volatility",
        "RL_Sharpe",
        "RL_MaxDD"
    ]
    df = df[[c for c in col_order if c in df.columns]]

    # Print results with tabulate
    print("\n=== BATCH RESULTS ===")
    print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))

def main():
    run_batch_experiments(num_runs=5)

if __name__ == "__main__":
    main()

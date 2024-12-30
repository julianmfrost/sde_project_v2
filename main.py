# main.py

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from factor_loader import FactorDataLoader
from dhr_model import DHRModel
from rl_agent import RLAgent

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def run_tests(data, factors):
    print("\nRunning preliminary tests...")

    # Test DHR
    print("\n1. Testing DHR Model Updates...")
    try:
        factor = 'Mkt-RF'
        initial_data = data[factor].values[:24]
        model = DHRModel(trend=True, seasonality=12)
        model.build_model()
        model.fit(initial_data)
        initial_forecast = model.forecast(steps=1)
        print(f"Initial forecast for {factor}: {initial_forecast}")
        print("DHR model test passed")
    except Exception as e:
        print(f"DHR model test failed: {str(e)}")

    # State Space Test
    print("\n2. Testing State Space Construction...")
    try:
        sample_forecasts = {factor: 0.01 for factor in factors}
        # Just a check
        state_indices = []
        for forecast in sample_forecasts.values():
            if forecast < -0.005:
                state_indices.append(0)
            elif forecast > 0.005:
                state_indices.append(2)
            else:
                state_indices.append(1)
        state = sum([idx * (3 ** i) for i, idx in enumerate(state_indices)])
        print(f"Sample state calculation: {state}")
        print("State space test passed")
    except Exception as e:
        print(f"State space test failed: {str(e)}")

    # Portfolio Weight Test
    print("\n3. Testing Portfolio Weight Updates...")
    try:
        weights = {factor: 1/len(factors) for factor in factors}
        print("Initial weights:", weights)
        weights[factors[0]] += 0.1
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        print("After adjustment:", weights)
        print("Weight adjustment test passed")
    except Exception as e:
        print(f"Weight adjustment test failed: {str(e)}")

    print("\nPreliminary tests completed.")
    return True

def compute_metrics(portfolio_values, rf_values, periods_per_year=12):
    pv = np.array(portfolio_values)
    returns = pv[1:] / pv[:-1] - 1.0
    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1)
    annualized_return = (1 + mean_r)**periods_per_year - 1
    annualized_vol = std_r * np.sqrt(periods_per_year)

    mean_rf = np.mean(rf_values)
    annualized_rf = (1 + mean_rf)**periods_per_year - 1

    if annualized_vol != 0:
        sharpe = (annualized_return - annualized_rf) / annualized_vol
    else:
        sharpe = np.nan

    running_max = np.maximum.accumulate(pv)
    drawdowns = (pv - running_max) / running_max
    max_dd = np.min(drawdowns)
    return annualized_return, annualized_vol, sharpe, max_dd

def compute_factor_momentum(factor_returns, window=3):
    # factor_returns: 2D array [T, num_factors]
    # We'll sum the last 'window' returns for each point and see if > 0
    T = factor_returns.shape[0]
    momentum = np.zeros((T, factor_returns.shape[1]))
    for t in range(T):
        if t < window:
            # Not enough data, set momentum as 0
            momentum[t, :] = 0.0
        else:
            # sum last 'window' returns
            window_sum = np.sum(factor_returns[t-window+1:t+1], axis=0)
            momentum[t, :] = window_sum
    return momentum

def main():
    try:
        data_file_path = r"C:\Users\frost\.vscode\SDE Project\raw data\factors dataset final.csv"
        loader = FactorDataLoader(data_file_path)
        data = loader.load_data()
        data = data.replace([np.inf, -np.inf], np.nan).dropna()

        factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']

        if run_tests(data, factors):
            print("\nProceeding with full training...")
        else:
            print("\nTests failed. Please fix issues before proceeding.")
            return

        split_index = int(len(data) * 0.8)
        training_data = data.iloc[:split_index].reset_index(drop=True)
        testing_data = data.iloc[split_index:].reset_index(drop=True)

        num_factors = len(factors)
        training_length = len(training_data)
        testing_length = len(testing_data)

        # Build and fit DHR models
        print("\nInitializing DHR models...")
        dhr_models = {}
        for factor in factors:
            print(f"Building model for {factor}...")
            y = training_data[factor].values
            dhr = DHRModel(trend=True, seasonality=12)
            dhr.build_model()
            dhr.fit(y)
            dhr_models[factor] = dhr

        # Precompute training forecasts
        factor_forecasts_training = np.zeros((training_length, num_factors))
        for f_idx, factor in enumerate(factors):
            model = dhr_models[factor]
            y = training_data[factor].values
            filtered_state_means, _ = model.kf.filter(y)
            for t in range(training_length):
                current_state = filtered_state_means[t]
                next_state = np.dot(model.kf.transition_matrices, current_state)
                forecast_val = np.dot(model.kf.observation_matrices, next_state)[0]
                factor_forecasts_training[t, f_idx] = forecast_val

        # Precompute testing forecasts (project forward)
        factor_forecasts_testing = np.zeros((testing_length, num_factors))
        for f_idx, factor in enumerate(factors):
            model = dhr_models[factor]
            y_train = training_data[factor].values
            filtered_state_means_train, _ = model.kf.filter(y_train)
            last_filtered_state = filtered_state_means_train[-1].copy()
            current_state = last_filtered_state
            for t in range(testing_length):
                current_state = np.dot(model.kf.transition_matrices, current_state)
                forecast_val = np.dot(model.kf.observation_matrices, current_state)[0]
                factor_forecasts_testing[t, f_idx] = forecast_val

        # Extract returns arrays
        train_factor_returns = training_data[factors].values
        test_factor_returns = testing_data[factors].values

        rf = training_data['RF'].values
        rf_test = testing_data['RF'].values
        mkt_rf = training_data['Mkt-RF'].values
        mkt_rf_test = testing_data['Mkt-RF'].values

        # Compute factor momentum signals (3-month window)
        # Note: Adjust window length as needed.
        factor_momentum_training = compute_factor_momentum(train_factor_returns, window=3)
        factor_momentum_testing = compute_factor_momentum(test_factor_returns, window=3)

        # RL Setup
        num_states_per_factor_forecast = 3  # Low/Neutral/High
        num_states_per_factor_momentum = 2  # Positive/Non-Positive
        # Combined states per factor = 3 * 2 = 6
        # Total states = 6^(num_factors), can be huge. This is just for demonstration.
        # Consider simplifying in practice or using a more advanced RL approach.

        num_states_per_factor = num_states_per_factor_forecast * num_states_per_factor_momentum
        num_states = (num_states_per_factor) ** num_factors
        num_actions = 3 ** num_factors  # same action space as before

        agent = RLAgent(state_space_size=num_states, action_space_size=num_actions)

        # Precompute action map
        action_map = []
        for a_idx in range(num_actions):
            temp = a_idx
            idx_list = []
            for _ in range(num_factors):
                idx_list.append(temp % 3)
                temp //= 3
            action_map.append(idx_list)

        action_effect = np.array([-0.1, 0.0, +0.1])

        # Functions to discretize forecast & momentum
        def discretize_forecast(val):
            if val < -0.005:
                return 0  # Low
            elif val > 0.005:
                return 2  # High
            else:
                return 1  # Neutral

        def discretize_momentum(val):
            # If last 3-month sum > 0 => positive, else non-positive
            return 1 if val > 0 else 0  # 0=Non-Pos,1=Pos to keep consistent indexing

        # Construct state index from both forecast and momentum
        # For each factor, factor_state = forecast_state + 3 * momentum_state
        # since forecast_state in {0,1,2} and momentum_state in {0,1}, factor_state ranges 0..5
        def get_factor_state(forecast_val, momentum_val):
            f_state = discretize_forecast(forecast_val)
            m_state = discretize_momentum(momentum_val)
            return f_state + num_states_per_factor_forecast * m_state

        # Precompute powers for state calculation
        powers = np.power(num_states_per_factor, np.arange(num_factors))

        # Initialize weights
        weight_array = np.ones(num_factors) / num_factors
        portfolio_values = []
        benchmark_values = []

        # Training Loop with benchmark-relative reward
        print("\nStarting training loop...")
        for t in range(training_length - 1):
            if t % 500 == 0:
                print(f"Processing training data point {t}/{training_length-2}")

            forecasts_array = factor_forecasts_training[t]
            momentum_array = factor_momentum_training[t]

            factor_states = []
            for f_idx in range(num_factors):
                f_state = get_factor_state(forecasts_array[f_idx], momentum_array[f_idx])
                factor_states.append(f_state)

            factor_states = np.array(factor_states)
            state = np.sum(factor_states * powers)

            action_idx = agent.choose_action(state)
            action_indices = action_map[action_idx]

            # Update weights
            weight_array += action_effect[np.array(action_indices)]
            weight_array = np.clip(weight_array, 0, 1)
            weight_array /= weight_array.sum()

            actual_returns = train_factor_returns[t+1]
            portfolio_return = np.sum(weight_array * actual_returns)
            benchmark_return = mkt_rf[t+1] + rf[t+1]

            # Benchmark-relative reward
            reward = portfolio_return - benchmark_return

            agent.update_q_table(state, action_idx, reward, state)
            agent.decay_exploration()

            if t == 0:
                portfolio_value = 1 + portfolio_return
                benchmark_value = 1 + benchmark_return
            else:
                portfolio_value = portfolio_values[-1] * (1 + portfolio_return)
                benchmark_value = benchmark_values[-1] * (1 + benchmark_return)

            portfolio_values.append(portfolio_value)
            benchmark_values.append(benchmark_value)

        ann_ret, ann_vol, sh_ratio, mdd = compute_metrics(portfolio_values, rf[1:])
        print("\nTraining Period Performance Metrics:")
        print(f"Annualized Return: {ann_ret:.2%}")
        print(f"Annualized Volatility: {ann_vol:.2%}")
        print(f"Sharpe Ratio: {sh_ratio:.2f}")
        print(f"Max Drawdown: {mdd:.2%}")

        plt.figure(figsize=(12,6))
        plt.plot(training_data['DATE'].iloc[1:], portfolio_values, label='RL Portfolio')
        plt.plot(training_data['DATE'].iloc[1:], benchmark_values, label='Benchmark (Mkt-RF+RF)', alpha=0.7)
        plt.title('Portfolio Value Over Time (Training)')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.show()

        # Testing Phase
        print("\nStarting testing phase (no re-fitting)...", flush=True)
        agent.reset_exploration()

        test_portfolio_values = []
        test_benchmark_values = []

        for t in range(testing_length - 1):
            if t % 500 == 0:
                print(f"Processing testing data point {t}/{testing_length-2}", flush=True)

            forecasts_array = factor_forecasts_testing[t]
            momentum_array = factor_momentum_testing[t]

            factor_states = []
            for f_idx in range(num_factors):
                f_state = get_factor_state(forecasts_array[f_idx], momentum_array[f_idx])
                factor_states.append(f_state)
            factor_states = np.array(factor_states)
            state = np.sum(factor_states * powers)

            action_idx = np.argmax(agent.q_table[state, :])
            action_indices = action_map[action_idx]

            weight_array += action_effect[np.array(action_indices)]
            weight_array = np.clip(weight_array, 0, 1)
            weight_array /= weight_array.sum()

            actual_returns = test_factor_returns[t+1]
            portfolio_return = np.sum(weight_array * actual_returns)
            benchmark_return = mkt_rf_test[t+1] + rf_test[t+1]

            if t == 0:
                starting_portfolio_val = portfolio_values[-1]
                starting_benchmark_val = benchmark_values[-1]
                portfolio_value = starting_portfolio_val * (1 + portfolio_return)
                benchmark_value = starting_benchmark_val * (1 + benchmark_return)
            else:
                portfolio_value = test_portfolio_values[-1] * (1 + portfolio_return)
                benchmark_value = test_benchmark_values[-1] * (1 + benchmark_return)

            test_portfolio_values.append(portfolio_value)
            test_benchmark_values.append(benchmark_value)

        ann_ret_test, ann_vol_test, sh_ratio_test, mdd_test = compute_metrics(test_portfolio_values, rf_test[1:])
        print("\nTesting Period Performance Metrics:")
        print(f"Annualized Return: {ann_ret_test:.2%}")
        print(f"Annualized Volatility: {ann_vol_test:.2%}")
        print(f"Sharpe Ratio: {sh_ratio_test:.2f}")
        print(f"Max Drawdown: {mdd_test:.2%}")

        plt.figure(figsize=(12,6))
        plt.plot(testing_data['DATE'].iloc[1:], test_portfolio_values, label='RL Portfolio')
        plt.plot(testing_data['DATE'].iloc[1:], test_benchmark_values, label='Benchmark (Mkt-RF+RF)', alpha=0.7)
        plt.title('Portfolio Value Over Time (Testing)')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

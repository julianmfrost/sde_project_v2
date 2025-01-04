import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

from factor_loader import FactorDataLoader
from dhr_model import DHRModel
from deep_rl_agent import DeepRLAgent

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def run_tests(data, factors):
    print("\nRunning preliminary tests...")
    print("\n1. Testing DHR Model Updates...")
    try:
        factor = factors[0]
        initial_data = data[factor].values[:24]
        model = DHRModel(trend=True, seasonality=12)
        model.build_model()
        model.fit(initial_data)
        initial_forecast = model.forecast(steps=1)
        print(f"Initial forecast for {factor}: {initial_forecast}")
        print("DHR model test passed")
    except Exception as e:
        print(f"DHR model test failed: {str(e)}")

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

def run_experiment(no_plot=False):
    """
    Runs one DHR+RL experiment, optionally suppressing plots if no_plot=True.
    Returns a dictionary of performance metrics from the testing phase.
    """

    try:
        # Adjust path as needed:
        data_file_path = r"/Users/julianfrost/Downloads/sde_project_v2-main/raw data/factors dataset final.csv"
        loader = FactorDataLoader(data_file_path)
        data = loader.load_data()
        data = data.replace([np.inf, -np.inf], np.nan).dropna()

        factors = ['SMB', 'HML', 'RMW', 'CMA']

        if not run_tests(data, factors):
            # If preliminary tests fail, return empty or handle as you wish
            return {}

        # Split data
        split_index = int(len(data) * 0.8)
        training_data = data.iloc[:split_index].reset_index(drop=True)
        testing_data = data.iloc[split_index:].reset_index(drop=True)

        num_factors = len(factors)
        num_states = 6 ** num_factors  # Forecast(3) * Momentum(2) = 6 states per factor
        num_actions = 3 ** num_factors  # 3 actions per factor

        # Build DHR models for each factor
        dhr_models = {}
        for factor in factors:
            y = training_data[factor].values
            dhr = DHRModel(trend=True, seasonality=12)
            dhr.build_model()
            dhr.fit(y)
            dhr_models[factor] = dhr

        # Prepare DHR forecasts for training
        training_length = len(training_data)
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

        # Prepare DHR forecasts for testing
        testing_length = len(testing_data)
        factor_forecasts_testing = np.zeros((testing_length, num_factors))
        for f_idx, factor in enumerate(factors):
            model = dhr_models[factor]
            y_train = training_data[factor].values
            filtered_state_means_train, _ = model.kf.filter(y_train)
            last_filtered_state = filtered_state_means_train[-1].copy()
            current_state = last_filtered_state
            test_y = testing_data[factor].values
            for t in range(testing_length):
                current_state = np.dot(model.kf.transition_matrices, current_state)
                forecast_val = np.dot(model.kf.observation_matrices, current_state)[0]
                factor_forecasts_testing[t, f_idx] = forecast_val

        train_factor_returns = training_data[factors].values
        test_factor_returns = testing_data[factors].values
        rf = training_data['RF'].values
        rf_test = testing_data['RF'].values

        def compute_momentum_states(returns_array, window=3):
            T = returns_array.shape[0]
            momentum_states = np.zeros((T, num_factors), dtype=int)
            for t in range(T):
                if t < window:
                    continue
                sum_3 = np.sum(returns_array[t - window + 1 : t + 1, :], axis=0)
                momentum_states[t, :] = (sum_3 > 0).astype(int)
            return momentum_states

        momentum_training = compute_momentum_states(train_factor_returns, window=3)
        momentum_testing = compute_momentum_states(test_factor_returns, window=3)

        def discretize_forecast(val):
            if val < -0.005:
                return 0
            elif val > 0.005:
                return 2
            else:
                return 1

        def factor_state_index(forecast_val, mom_val):
            # forecast_val in {0,1,2}, mom_val in {0,1}
            return forecast_val + 3 * mom_val

        def state_to_index(factor_states):
            index = 0
            power = 1
            for s in factor_states:
                index += s * power
                power *= 6
            return index

        def action_idx_to_changes(action_idx):
            changes = []
            temp = action_idx
            for _ in range(num_factors):
                a = temp % 3
                temp //= 3
                changes.append(a)
            return changes

        agent = DeepRLAgent(
            state_space_size=num_states,
            action_space_size=num_actions,
            learning_rate=0.0005
        )
        action_effect = np.array([-0.1, 0.0, 0.1])

        roll_window = 30
        training_returns_deque = deque(maxlen=roll_window)

        weight_array = np.ones(num_factors) / num_factors
        portfolio_values = []

        # Factor BH / eq-weight for training (just for internal reference/plots)
        train_factor_BH_values = np.ones((training_length, num_factors))
        train_eq_weight_values = np.ones(training_length)

        # ---- Training Loop ----
        for t in range(training_length - 1):
            factor_states = []
            for f_idx in range(num_factors):
                f_cast = discretize_forecast(factor_forecasts_training[t, f_idx])
                f_mom = momentum_training[t, f_idx]
                f_state_idx = factor_state_index(f_cast, f_mom)
                factor_states.append(f_state_idx)

            state = state_to_index(factor_states)
            action_idx = agent.choose_action(state)
            action_decoded = action_idx_to_changes(action_idx)

            # Adjust weights
            for i, a_val in enumerate(action_decoded):
                weight_array[i] += action_effect[a_val]
            weight_array = np.clip(weight_array, 0, 1)
            weight_array /= weight_array.sum()

            actual_returns = train_factor_returns[t + 1]
            portfolio_return = np.sum(weight_array * actual_returns)

            # BH updates (for reference)
            for f_idx in range(num_factors):
                train_factor_BH_values[t + 1, f_idx] = train_factor_BH_values[t, f_idx] * (1 + actual_returns[f_idx])
            eq_weight_return = np.mean(actual_returns)
            train_eq_weight_values[t + 1] = train_eq_weight_values[t] * (1 + eq_weight_return)

            if t == 0:
                portfolio_value = 1 + portfolio_return
            else:
                portfolio_value = portfolio_values[-1] * (1 + portfolio_return)
            portfolio_values.append(portfolio_value)

            training_returns_deque.append(portfolio_return)
            if len(training_returns_deque) == roll_window:
                vol = np.std(training_returns_deque, ddof=1)
            else:
                vol = 0.0

            # Reward: RL port minus eq-weight minus volatility penalty
            reward = (portfolio_return - eq_weight_return) - 0.01 * vol

            # Next state
            if t + 1 < training_length - 1:
                next_factor_states = []
                for f_idx in range(num_factors):
                    f_cast_next = discretize_forecast(factor_forecasts_training[t + 1, f_idx])
                    f_mom_next = momentum_training[t + 1, f_idx]
                    next_factor_states.append(factor_state_index(f_cast_next, f_mom_next))
                next_state = state_to_index(next_factor_states)
            else:
                next_state = None

            agent.update_q_table(state, action_idx, reward, next_state)

        # Training Metrics
        ann_ret_train, ann_vol_train, sh_ratio_train, mdd_train = compute_metrics(portfolio_values, rf[1:])

        if not no_plot:
            print("\nTraining Period Performance Metrics:")
            print(f"Annualized Return: {ann_ret_train:.2%}")
            print(f"Annualized Volatility: {ann_vol_train:.2%}")
            print(f"Sharpe Ratio: {sh_ratio_train:.2f}")
            print(f"Max Drawdown: {mdd_train:.2%}")

            # Plot training
            plt.figure(figsize=(12, 6))
            plt.plot(training_data['DATE'].iloc[1:], portfolio_values, label='DHR-RL Portfolio (Train)')
            for f_idx, factor in enumerate(factors):
                plt.plot(training_data['DATE'].iloc[1:], train_factor_BH_values[1:, f_idx], label=f"{factor} BH")
            plt.plot(training_data['DATE'].iloc[1:], train_eq_weight_values[1:], label='Equal-Weighted BH', linestyle='--')
            plt.title('Portfolio Value - Training')
            plt.legend()
            plt.show()

        # ---- Testing Phase ----
        agent.reset_exploration()
        agent.exploration_rate = 0.01

        test_portfolio_values = []
        test_factor_BH_values = np.ones((testing_length, num_factors))
        test_eq_weight_values = np.ones(testing_length)
        test_returns_deque = deque(maxlen=roll_window)

        test_weight_array = np.ones(num_factors) / num_factors

        for t in range(testing_length - 1):
            test_factor_states = []
            for f_idx in range(num_factors):
                f_cast_test = discretize_forecast(factor_forecasts_testing[t, f_idx])
                f_mom_test = momentum_testing[t, f_idx]
                test_factor_states.append(factor_state_index(f_cast_test, f_mom_test))

            state_test = state_to_index(test_factor_states)
            action_idx = agent.choose_action(state_test)
            action_decoded = action_idx_to_changes(action_idx)

            for i, a_val in enumerate(action_decoded):
                test_weight_array[i] += action_effect[a_val]
            test_weight_array = np.clip(test_weight_array, 0, 1)
            test_weight_array /= test_weight_array.sum()

            actual_returns = test_factor_returns[t + 1]
            portfolio_return = np.sum(test_weight_array * actual_returns)

            # BH updates (for reference)
            for f_idx in range(num_factors):
                test_factor_BH_values[t + 1, f_idx] = test_factor_BH_values[t, f_idx] * (1 + actual_returns[f_idx])
            eq_weight_return_test = np.mean(actual_returns)
            test_eq_weight_values[t + 1] = test_eq_weight_values[t] * (1 + eq_weight_return_test)

            if t == 0:
                portfolio_value_test = 1.0 * (1 + portfolio_return)
            else:
                portfolio_value_test = test_portfolio_values[-1] * (1 + portfolio_return)
            test_portfolio_values.append(portfolio_value_test)

            test_returns_deque.append(portfolio_return)

        # Testing Metrics
        ann_ret_test, ann_vol_test, sh_ratio_test, mdd_test = compute_metrics(test_portfolio_values, rf_test[1:])

        if not no_plot:
            print("\nTesting Period Performance Metrics:")
            print(f"Annualized Return: {ann_ret_test:.2%}")
            print(f"Annualized Volatility: {ann_vol_test:.2%}")
            print(f"Sharpe Ratio: {sh_ratio_test:.2f}")
            print(f"Max Drawdown: {mdd_test:.2%}")

            # Plot testing
            plt.figure(figsize=(12, 6))
            plt.plot(testing_data['DATE'].iloc[1:], test_portfolio_values, label='DHR-RL Portfolio (Test)')
            for f_idx, factor in enumerate(factors):
                plt.plot(testing_data['DATE'].iloc[1:], test_factor_BH_values[1:, f_idx], label=f"{factor} BH")
            plt.plot(testing_data['DATE'].iloc[1:], test_eq_weight_values[1:], label='Equal-Weighted BH', linestyle='--')
            plt.title('Portfolio Value - Testing')
            plt.legend()
            plt.show()

        # Return dictionary with final testing metrics
        results_dict = {
            "RL_AnnualizedReturn": ann_ret_test,
            "RL_Volatility": ann_vol_test,
            "RL_Sharpe": sh_ratio_test,
            "RL_MaxDD": mdd_test
        }
        return results_dict

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return {}

def main():
    # Single-run scenario (show plots)
    res = run_experiment(no_plot=False)
    if res:
        print("\nFinal Testing Metrics:")
        for k,v in res.items():
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()

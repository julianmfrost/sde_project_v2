#main_deep_rl_experiments.py

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

from factor_loader import FactorDataLoader
from dhr_model import DHRModel
from deep_rl_agent import DeepRLAgent

#test2

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

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
    Runs one DHR+RL experiment, returning a dictionary that includes:
    - Training RL and EW metrics
    - Testing RL and EW metrics
    If no_plot=True, it won't show any matplotlib plots.
    """

    try:
        # Adjust path if needed:
        data_file_path = r"/Users/julianfrost/Downloads/sde_project_v2-main/raw data/factors dataset final.csv"
        loader = FactorDataLoader(data_file_path)
        data = loader.load_data()
        data = data.replace([np.inf, -np.inf], np.nan).dropna()

        factors = ['SMB', 'HML', 'RMW', 'CMA']

        # Preliminary test
        if not run_tests(data, factors):
            return {}

        # 80/20 split as an example
        split_index = int(len(data) * 0.8)
        training_data = data.iloc[:split_index].reset_index(drop=True)
        testing_data = data.iloc[split_index:].reset_index(drop=True)

        num_factors = len(factors)
        num_states = 6 ** num_factors  # Forecast(3) * Momentum(2) = 6 states per factor
        num_actions = 3 ** num_factors # 3 actions per factor

        # ---- Build DHR models for each factor
        dhr_models = {}
        for factor in factors:
            y = training_data[factor].values
            dhr = DHRModel(trend=True, seasonality=12)
            dhr.build_model()
            dhr.fit(y)
            dhr_models[factor] = dhr

        # ---- Prepare DHR forecasts (training)
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

        # ---- Prepare DHR forecasts (testing)
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
        rf_train = training_data['RF'].values
        rf_test = testing_data['RF'].values

        # Momentum signals (3-month)
        def compute_momentum_states(returns_array, window=3):
            T = returns_array.shape[0]
            momentum_states = np.zeros((T, num_factors), dtype=int)
            for t in range(T):
                if t < window:
                    continue
                sum_3 = np.sum(returns_array[t-window+1:t+1, :], axis=0)
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

        def factor_state_index(f_cast, m_val):
            return f_cast + 3*m_val

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
                temp //=3
                changes.append(a)
            return changes

        agent = DeepRLAgent(
            state_space_size=num_states,
            action_space_size=num_actions,
            learning_rate=0.0005
        )
        action_effect = np.array([-0.1, 0.0, 0.1])

        # ---- Training Phase
        training_returns_deque = deque(maxlen=30)
        rl_weights_train = np.ones(num_factors)/num_factors
        rl_port_values_train = []

        # Track eq-weight portfolio for training
        eq_port_values_train = [1.0]  # start at 1
        eq_cumprod = 1.0

        # We also store a parallel array of len(training_length)
        eq_array_train = np.ones(training_length)

        for t in range(training_length-1):
            # Build state
            factor_states = []
            for f_idx in range(num_factors):
                f_val = discretize_forecast(factor_forecasts_training[t,f_idx])
                m_val = momentum_training[t,f_idx]
                factor_states.append(factor_state_index(f_val,m_val))

            state = state_to_index(factor_states)
            action_idx = agent.choose_action(state)
            action_decoded = action_idx_to_changes(action_idx)

            # Adjust RL weights
            for i,a_val in enumerate(action_decoded):
                rl_weights_train[i] += action_effect[a_val]
            rl_weights_train = np.clip(rl_weights_train,0,1)
            rl_weights_train /= rl_weights_train.sum()

            # RL portfolio return
            actual_returns = train_factor_returns[t+1]
            rl_ret = np.sum(rl_weights_train * actual_returns)

            if t==0:
                rl_val = 1+rl_ret
            else:
                rl_val = rl_port_values_train[-1] * (1+rl_ret)

            rl_port_values_train.append(rl_val)

            # eq-weight for training
            eq_ret_train = np.mean(actual_returns)
            eq_cumprod *= (1 + eq_ret_train)
            eq_array_train[t+1] = eq_cumprod

            # Rolling vol for reward
            training_returns_deque.append(rl_ret)
            if len(training_returns_deque)==30:
                vol = np.std(training_returns_deque,ddof=1)
            else:
                vol=0.0

            # Reward
            reward = (rl_ret - eq_ret_train) - 0.01*vol

            # Next state
            if t+1 < training_length-1:
                factor_states_next=[]
                for f_idx in range(num_factors):
                    f_val_next = discretize_forecast(factor_forecasts_training[t+1,f_idx])
                    m_val_next = momentum_training[t+1,f_idx]
                    factor_states_next.append(factor_state_index(f_val_next,m_val_next))
                next_state = state_to_index(factor_states_next)
            else:
                next_state=None

            agent.update_q_table(state,action_idx,reward,next_state)

        # Compute training stats
        ann_ret_train_rl, ann_vol_train_rl, sh_train_rl, mdd_train_rl = compute_metrics(rl_port_values_train, rf_train[1:])
        ann_ret_train_eq, ann_vol_train_eq, sh_train_eq, mdd_train_eq = compute_metrics(eq_array_train, rf_train)

        # ---- Testing Phase
        agent.reset_exploration()
        agent.exploration_rate=0.01

        rl_weights_test = np.ones(num_factors)/num_factors
        rl_port_values_test=[]
        eq_array_test= np.ones(testing_length)
        eq_cumprod_test=1.0

        test_returns_deque=deque(maxlen=30)

        for t in range(testing_length-1):
            factor_states_test=[]
            for f_idx in range(num_factors):
                f_val_test = discretize_forecast(factor_forecasts_testing[t,f_idx])
                m_val_test = momentum_testing[t,f_idx]
                factor_states_test.append(factor_state_index(f_val_test,m_val_test))
            state_test = state_to_index(factor_states_test)

            action_idx=agent.choose_action(state_test)
            action_decoded = action_idx_to_changes(action_idx)

            for i,a_val in enumerate(action_decoded):
                rl_weights_test[i]+=action_effect[a_val]
            rl_weights_test = np.clip(rl_weights_test,0,1)
            rl_weights_test/=rl_weights_test.sum()

            actual_returns_test = test_factor_returns[t+1]
            rl_ret_test = np.sum(rl_weights_test*actual_returns_test)

            if t==0:
                val_test = 1+rl_ret_test
            else:
                val_test= rl_port_values_test[-1]*(1+rl_ret_test)
            rl_port_values_test.append(val_test)

            # eq-weight
            eq_ret_test = np.mean(actual_returns_test)
            eq_cumprod_test *= (1 + eq_ret_test)
            eq_array_test[t+1] = eq_cumprod_test

            test_returns_deque.append(rl_ret_test)

        # final metrics
        ann_ret_test_rl, ann_vol_test_rl, sh_test_rl, mdd_test_rl = compute_metrics(rl_port_values_test, rf_test[1:])
        ann_ret_test_eq, ann_vol_test_eq, sh_test_eq, mdd_test_eq = compute_metrics(eq_array_test, rf_test)

        # If no_plot=False, show training+testing charts, else skip
        if not no_plot:
            # Print training metrics
            print("\nTraining RL vs. EW:")
            print(f"RL Annualized Return: {ann_ret_train_rl:.2%}, Sharpe: {sh_train_rl:.2f}, MaxDD: {mdd_train_rl:.2%}")
            print(f"EW Annualized Return: {ann_ret_train_eq:.2%}, Sharpe: {sh_train_eq:.2f}, MaxDD: {mdd_train_eq:.2%}")

            plt.figure(figsize=(12,5))
            plt.plot(rl_port_values_train, label='RL (Train)')
            plt.plot(eq_array_train, label='EW (Train)')
            plt.title("Training Comparison")
            plt.legend()
            plt.show()

            # Print testing metrics
            print("\nTesting RL vs. EW:")
            print(f"RL Annualized Return: {ann_ret_test_rl:.2%}, Sharpe: {sh_test_rl:.2f}, MaxDD: {mdd_test_rl:.2%}")
            print(f"EW Annualized Return: {ann_ret_test_eq:.2%}, Sharpe: {sh_test_eq:.2f}, MaxDD: {mdd_test_eq:.2%}")

            plt.figure(figsize=(12,5))
            plt.plot(rl_port_values_test, label='RL (Test)')
            plt.plot(eq_array_test, label='EW (Test)')
            plt.title("Testing Comparison")
            plt.legend()
            plt.show()

        # Return dictionary with both training & testing stats for RL & EW
        return {
            # Training RL
            "Train_RL_AnnRet": ann_ret_train_rl,
            "Train_RL_Vol": ann_vol_train_rl,
            "Train_RL_Sharpe": sh_train_rl,
            "Train_RL_MaxDD": mdd_train_rl,

            # Training EW
            "Train_EW_AnnRet": ann_ret_train_eq,
            "Train_EW_Vol": ann_vol_train_eq,
            "Train_EW_Sharpe": sh_train_eq,
            "Train_EW_MaxDD": mdd_train_eq,

            # Testing RL
            "Test_RL_AnnRet": ann_ret_test_rl,
            "Test_RL_Vol": ann_vol_test_rl,
            "Test_RL_Sharpe": sh_test_rl,
            "Test_RL_MaxDD": mdd_test_rl,

            # Testing EW
            "Test_EW_AnnRet": ann_ret_test_eq,
            "Test_EW_Vol": ann_vol_test_eq,
            "Test_EW_Sharpe": sh_test_eq,
            "Test_EW_MaxDD": mdd_test_eq
        }

    except Exception as e:
        print(f"Error in run_experiment: {str(e)}")
        return {}

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


def main():
    # Single-run scenario with plots
    results = run_experiment(no_plot=False)
    if results:
        print("\nFinal Testing Metrics:")
        for k, v in results.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

if __name__ == "__main__":
    main()

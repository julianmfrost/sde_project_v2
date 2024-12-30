# rl_agent.py
import numpy as np
import random

class RLAgent:
    def __init__(self, state_space_size, action_space_size, learning_rate=0.1, discount_factor=0.95,
                 exploration_rate=1.0, exploration_decay=0.99, min_exploration_rate=0.01):
        """
        Initialize the Reinforcement Learning agent.

        Parameters:
        - state_space_size (int): Number of possible states in the environment.
        - action_space_size (int): Number of possible actions the agent can take.
        - learning_rate (float): Learning rate for Q-value updates.
        - discount_factor (float): Discount factor for future rewards (gamma).
        - exploration_rate (float): Initial exploration rate (epsilon).
        - exploration_decay (float): Decay factor for exploration rate.
        - min_exploration_rate (float): Minimum exploration rate.
        """
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

        # Initialize Q-table with zeros
        self.q_table = np.zeros((state_space_size, action_space_size))

    def choose_action(self, state):
        """
        Choose an action based on the exploration-exploitation tradeoff.

        Parameters:
        - state (int): The current state.

        Returns:
        - action (int): The selected action.
        """
        if random.uniform(0, 1) < self.exploration_rate:
            # Explore: Choose a random action
            return random.randint(0, self.action_space_size - 1)
        else:
            # Exploit: Choose the action with the highest Q-value for the current state
            return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state):
        """
        Update the Q-value for the given state-action pair.

        Parameters:
        - state (int): The current state.
        - action (int): The action taken.
        - reward (float): The reward received.
        - next_state (int): The next state.
        """
        best_next_action = np.argmax(self.q_table[next_state, :])
        td_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]

        # Update Q-value
        self.q_table[state, action] += self.learning_rate * td_error

    def decay_exploration(self):
        """
        Decay the exploration rate after each episode.
        """
        self.exploration_rate = max(self.min_exploration_rate,
                                    self.exploration_rate * self.exploration_decay)

    def reset_exploration(self):
        """
        Reset the exploration rate to its initial value.
        """
        self.exploration_rate = 1.0
        
# deep_rl_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DeepRLAgent:
    def __init__(self,
                 state_space_size,
                 action_space_size,
                 learning_rate=0.001,
                 discount_factor=0.95,
                 exploration_rate=1.0,
                 exploration_decay=0.999,
                 min_exploration_rate=0.01,
                 batch_size=32,
                 replay_buffer_size=10000,
                 update_target_every=500):
        """
        A simple DQN-based agent. 
        """
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.batch_size = batch_size
        self.update_target_every = update_target_every

        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.steps_done = 0

        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNetwork(self.state_space_size, self.action_space_size).to(self.device)
        self.target_net = DQNNetwork(self.state_space_size, self.action_space_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def one_hot_state(self, state):
        # Convert integer state to one-hot tensor
        state_vec = np.zeros(self.state_space_size, dtype=np.float32)
        state_vec[state] = 1.0
        return torch.tensor(state_vec, device=self.device).unsqueeze(0)

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if random.random() < self.exploration_rate:
            return random.randint(0, self.action_space_size - 1)
        else:
            with torch.no_grad():
                s = self.one_hot_state(state)
                q_values = self.policy_net(s)
                return int(torch.argmax(q_values).item())

    def remember(self, state, action, reward, next_state):
        self.replay_buffer.append((state, action, reward, next_state))

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states_t = torch.cat([self.one_hot_state(s) for s in states])
        actions_t = torch.tensor(actions, device=self.device, dtype=torch.int64).unsqueeze(-1)
        rewards_t = torch.tensor(rewards, device=self.device, dtype=torch.float32).unsqueeze(-1)
        non_final_mask = torch.tensor([ns is not None for ns in next_states], device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([self.one_hot_state(ns) for ns in next_states if ns is not None])

        current_q_values = self.policy_net(states_t).gather(1, actions_t)

        next_q_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            if non_final_mask.any():
                next_q_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        target_q_values = rewards_t + (self.discount_factor * next_q_values.unsqueeze(-1))

        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_exploration(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

    def update_q_table(self, state, action, reward, next_state):
        # In DQN, we don't directly update Q-values, we store transitions and learn from replay
        self.remember(state, action, reward, next_state)
        self.replay()
        self.decay_exploration()

    def reset_exploration(self):
        self.exploration_rate = 1.0

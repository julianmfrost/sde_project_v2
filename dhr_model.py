# dhr_model.py
import numpy as np
from pykalman import KalmanFilter

class DHRModel:
    def __init__(self, trend=True, seasonality=12):
        self.trend = trend
        self.seasonality = seasonality
        self.kf = None
        self.observations = None  # Add this line

    def build_model(self):
        # Build the state-space representation
        transition_matrix = self._build_transition_matrix()
        observation_matrix = np.zeros((1, transition_matrix.shape[0]))
        observation_matrix[0, 0] = 1  # Only the first state contributes to the observation

        # Initialize the Kalman Filter
        self.kf = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            initial_state_mean=np.zeros(transition_matrix.shape[0]),
            initial_state_covariance=np.eye(transition_matrix.shape[0]),
            transition_covariance=np.eye(transition_matrix.shape[0]) * 0.1,
            observation_covariance=np.array([[0.1]])
        )

    def _build_transition_matrix(self):
        # Build the transition matrix
        state_size = 1 + (2 * self.seasonality if self.seasonality else 0)
        transition_matrix = np.eye(state_size)

        # Add seasonal harmonics
        for i in range(1, self.seasonality + 1):
            idx_cos = 2 * i - 1
            idx_sin = 2 * i
            angle = 2 * np.pi * i / self.seasonality
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            transition_matrix[idx_cos, idx_cos] = cos_angle
            transition_matrix[idx_cos, idx_sin] = -sin_angle
            transition_matrix[idx_sin, idx_cos] = sin_angle
            transition_matrix[idx_sin, idx_sin] = cos_angle

        return transition_matrix

    def fit(self, y):
        self.observations = y
        self.kf = self.kf.em(y, n_iter=20)
        self.observations = y  # Set it again after EM

    def forecast(self, steps=1):
        if self.observations is None:
            raise ValueError("Model must be fitted before forecasting")
        filtered_state_means, _ = self.kf.filter(self.observations)  # Use self.observations instead
        forecasts = []
        current_state = filtered_state_means[-1]
        for _ in range(steps):
            current_state = np.dot(self.kf.transition_matrices, current_state)
            forecast = np.dot(self.kf.observation_matrices, current_state)
            forecasts.append(forecast[0])
        return np.array(forecasts)

    def smooth(self, y):
        state_means, _ = self.kf.smooth(y)
        return state_means
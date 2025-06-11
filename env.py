import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PortfolioEnv(gym.Env):
    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int = 30, # (30, 10, 6)
        transaction_cost: float = 0.0001,
        risk_free_rate: float = 0.02,
        initial_balance: float = 1_000_000,
        n_assets: int = 10,
        use_sortino: bool = True
    ):
        super(PortfolioEnv, self).__init__()

        self.data = data
        self.features = data.columns.levels[0]
        self.assets = data.columns.levels[1]
        self.n_assets = n_assets
        self.n_features = len(self.features)
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        self.initial_balance = initial_balance
        self.use_sortino = use_sortino
        self.max_steps = len(self.data) - 1

        # Action space is discrete with 3 actions: 0 = sell, 1 = hold, 2 = buy
        self.action_space = spaces.MultiDiscrete([3] * self.n_assets)
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.n_assets, self.n_features),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = self.window_size
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.prev_weights = self.weights.copy()
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.net_worths = [self.initial_balance]

        observation = self._get_observation()
        info = {
            "balance": self.balance,
            "weights": self.weights,
            "portfolio_value": self.portfolio_value,
        }
        return observation, info

    def step(self, action):
        weight_change = np.zeros_like(self.weights)
        for i in range(self.n_assets):
            if action[i] == 0:  # Sell
                weight_change[i] = -0.1
            elif action[i] == 1:  # Hold
                weight_change[i] = 0.0
            elif action[i] == 2:  # Buy
                weight_change[i] = 0.1

        # Apply noise to encourage exploration
        noise = np.random.normal(0, 0.005 * (1 - self.current_step / self.max_steps), size=self.n_assets)
        weight_change += noise

        # Update weights and normalize
        self.weights = np.clip(self.weights + weight_change, 0, 1)
        total_weight = np.sum(self.weights)
        if total_weight > 0:
            self.weights /= total_weight
        else:
            self.weights = np.ones_like(self.weights) / self.n_assets

        log_returns = self.data.xs("log_return", axis=1, level=0).iloc[self.current_step].values
        portfolio_return = np.dot(self.weights, log_returns)

        # Cost proportional to wealth
        cost = self.transaction_cost * self.portfolio_value * np.sum(np.abs(self.weights - self.prev_weights))

        # Update portfolio value
        self.portfolio_value *= np.exp(portfolio_return)
        self.portfolio_value -= cost
        self.portfolio_value = max(0, self.portfolio_value)

        # Base reward: prioritize return over cost
        cost_penalty = (cost / self.portfolio_value if self.portfolio_value > 0 else 0)
        reward = portfolio_return * 1.5 -  cost_penalty * 0.5

        # Bonus for low-cost profitable trades
        if portfolio_return > 0 and cost < 0.001 * self.portfolio_value:
            reward += 0.05

        # Risk-adjusted bonus (Sortino or Sharpe)
        sortino_or_sharpe = 0
        if self.current_step >= self.window_size:
            log_return_window = self.data.xs("log_return", axis=1, level=0).iloc[self.current_step - self.window_size:self.current_step]
            window_returns = log_return_window.dot(self.weights)
            rf_daily = self.risk_free_rate / 252
            excess_returns = window_returns - rf_daily

            if self.use_sortino:
                downside_returns = excess_returns[excess_returns < 0]
                downside_dev = np.std(downside_returns) if len(downside_returns) > 0 else 0
                if downside_dev > 0:
                    sortino_or_sharpe = np.mean(excess_returns) / downside_dev
            else:
                std_dev = np.std(window_returns)
                if std_dev > 0:
                    sortino_or_sharpe = np.mean(excess_returns) / std_dev

            reward += sortino_or_sharpe * 1.5  # Give more weight to risk-adjusted reward

        # Apply smooth reward and cap it
        reward = np.tanh(reward)
        reward = np.clip(reward, -1, 1)

        # Update state
        self.net_worths.append(self.portfolio_value)
        self.prev_weights = self.weights.copy()
        self.current_step += 1

        # Termination
        terminated = self.portfolio_value <= 0 or self.current_step >= len(self.data) - 1
        truncated = False

        obs = self._get_observation()
        info = {
            "portfolio_return": portfolio_return,
            "transaction_cost": cost,
            "weights": self.weights,
            "risk_adjusted_bonus": sortino_or_sharpe,
            "net_worth": self.portfolio_value
        }

        if self.current_step % 1000 == 0:
            print(f"Step: {self.current_step}, Reward: {reward:.4f}, Return: {portfolio_return:.4f}, Cost: {cost:.2f}, Risk Adj: {sortino_or_sharpe:.4f}")

        return obs, reward, terminated, truncated, info


    def _get_observation(self):
        #print("Current Step:", self.current_step)
        #print("Data Shape:", self.data.shape)
        
        obs_window = self.data.iloc[self.current_step - self.window_size:self.current_step]
        #print("Observation Window Shape:", obs_window.shape)
        
        if obs_window.empty:
            raise ValueError("Observation window is empty at step {}".format(self.current_step))

        obs_array = obs_window.stack(level=1, future_stack=True).values.reshape(self.window_size, self.n_assets, self.n_features)
        return obs_array
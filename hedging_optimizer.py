"""
Dynamic Hedging Optimizer using Reinforcement Learning
Implements advanced RL-based portfolio rebalancing and options pricing models
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.model_selection import train_test_split
import gym
from gym import spaces
from gym.utils import seeding
import torch
import torch.nn as nn
import torch.optim as optim


class DynamicHedgingEnv(gym.Env):
    """
    Custom Environment for RL-based Dynamic Hedging
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, data: pd.DataFrame, initial_cash: float):
        super(DynamicHedgingEnv, self).__init__()

        self.data = data
        self.initial_cash = initial_cash
        self.current_step = None
        self.done = None
        self.actions = None
        self.cash = None
        self.portfolio_value_history = []

        # Assume action space corresponds to [-1, 1] for each asset, including cash
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.data.columns),), dtype=np.float32)
        
        # Observations include current portfolio, current market prices
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(len(self.data.columns)+1,), dtype=np.float32)
            
    def reset(self):
        self.cash = self.initial_cash
        self.current_step = 0
        self.done = False
        self.portfolio_value_history = []        
        
        return self._get_observation()

    def _get_observation(self):
        # Portfolio and current prices
        return np.concatenate(([self.cash], self.data.iloc[self.current_step].values))

    def step(self, action: np.ndarray):
        self._trade(action)

        self.current_step += 1

        if self.current_step >= len(self.data):
            self.done = True

        reward = self._calculate_reward()
        obs = self._get_observation()

        return obs, reward, self.done, {}

    def _trade(self, action):
        # Calculate the allocation
        total_value = self.cash + np.sum(self.actions * self.data.iloc[self.current_step])
        allocation = total_value * action / np.sum(np.abs(action))

        # Update cash and actions
        self.cash = total_value - np.sum(allocation)
        self.actions = allocation / self.data.iloc[self.current_step]

    def _calculate_reward(self):
        # Reward could be Sharpe Ratio, Sortino, or simple return
        portfolio_value = self.cash + np.sum(self.actions * self.data.iloc[self.current_step])
        self.portfolio_value_history.append(portfolio_value)
        reward = portfolio_value 
        if self.current_step > 1:
            past_value = self.portfolio_value_history[-2]
            return reward - past_value  # Absolute change
        return reward


class HedgingPolicy(nn.Module):
    """Simple MLP Policy for Dynamic Hedging"""
    def __init__(self, input_dims: int, action_dims: int):
        super(HedgingPolicy, self).__init__()

        self.fc1 = nn.Linear(input_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dims)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


def train_rl_hedging_model(data: pd.DataFrame, initial_cash: float, n_episodes: int = 1000):
    env = DynamicHedgingEnv(data=data, initial_cash=initial_cash)
    policy = HedgingPolicy(input_dims=env.observation_space.shape[0], action_dims=env.action_space.shape[0])
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    for episode in range(n_episodes):
        observation = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = policy(torch.tensor(observation, dtype=torch.float32)).detach().numpy()
            observation, reward, done, _ = env.step(action)
            total_reward += reward

            optimizer.zero_grad()
            loss = -reward 
            loss.backward()
            optimizer.step()

        print(f'Episode {episode + 1}/{n_episodes}, Total Reward: {total_reward}')


# Advanced Options Pricing Models

from scipy.stats import norm

class OptionPricingModel:
    """Advanced options pricing with Black-Scholes and Monte Carlo simulations"""
    def __init__(self, volatility: float, risk_free_rate: float):
        self.volatility = volatility
        self.risk_free_rate = risk_free_rate

    def black_scholes(self, S: float, K: float, T: float, option_type: str = 'call'):
        """Calculate option price using Black-Scholes formula"""
        d1 = (np.log(S / K) + (self.risk_free_rate + 0.5 * self.volatility ** 2) * T) / (self.volatility * np.sqrt(T))
        d2 = d1 - self.volatility * np.sqrt(T)

        if option_type == 'call':
            option_price = S * norm.cdf(d1) - K * np.exp(-self.risk_free_rate * T) * norm.cdf(d2)
        elif option_type == 'put':
            option_price = K * np.exp(-self.risk_free_rate * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        return option_price

    def monte_carlo(self, S: float, K: float, T: float, option_type: str = 'call', n_simulations: int = 10000):
        """Calculate option price using Monte Carlo simulation"""
        payoffs = []
        discount_factor = np.exp(-self.risk_free_rate * T)

        for _ in range(n_simulations):
            Z = np.random.standard_normal()
            ST = S * np.exp((self.risk_free_rate - 0.5 * self.volatility ** 2) * T + self.volatility * np.sqrt(T) * Z)

            if option_type == 'call':
                payoff = max(ST - K, 0)
            elif option_type == 'put':
                payoff = max(K - ST, 0)
            else:
                raise ValueError("option_type must be 'call' or 'put'")

            payoffs.append(payoff)

        option_price = discount_factor * np.mean(payoffs)

        return option_price


# Example usage and testing
if __name__ == "__main__":
    from data_pipeline import ClimateDataPipeline

    # Create sample data
    pipeline = ClimateDataPipeline()
    tickers = ['TSLA', 'XOM', 'AAPL', 'NEE']
    data = pipeline.create_integrated_dataset(tickers, '2023-01-01', '2024-01-01')

    # Train RL hedging model
    train_rl_hedging_model(data=data, initial_cash=1000000, n_episodes=10)

    # Simple option pricing
    option_model = OptionPricingModel(volatility=0.2, risk_free_rate=0.01)
    call_price = option_model.black_scholes(S=150, K=148, T=1, option_type='call')
    put_price = option_model.monte_carlo(S=150, K=148, T=1, option_type='put')

    print(f"Call Price (Black-Scholes): {call_price}")
    print(f"Put Price (Monte Carlo): {put_price}")

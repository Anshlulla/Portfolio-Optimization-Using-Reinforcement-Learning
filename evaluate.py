import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import PortfolioEnv
from dataloader import load_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load test data (use different data or split the original data into train/test)
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
log_returns = load_data(tickers, start="2023-04-10", end="2025-04-10")  # Test data
print(log_returns.shape)
print(log_returns.head())

# Create environment
env = PortfolioEnv(log_returns, window_size=30, n_assets=5)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=False, norm_reward=True)

# Load the trained PPO model
model = PPO.load("ppo_portfolio_model_5n", env=env)

# Evaluate the model
def evaluate_model(model, env, episodes=25, transaction_cost=0.0001):
    rewards = []
    portfolio_values = []
    transaction_costs = []
    cumulative_returns = []
    
    # Data for visualization
    episode_rewards = []
    episode_transaction_costs = []
    episode_cumulative_returns = []
    episode_sharpe_ratios = []
    episode_sortino_ratios = []
    
    for episode in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        total_value = 1  # Assume initial portfolio value is 1 (100% of capital)
        prev_action = None
        total_transaction_cost = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            total_reward += reward
            
            # Check if a transaction occurred (buy/sell)
            if prev_action is not None and np.any(action != prev_action):  # action changed (buy/sell)
                # Calculate transaction cost: For simplicity, assume cost per transaction
                cost = transaction_cost * np.abs(total_value)  # proportional to portfolio value
                total_transaction_cost += cost
                
            prev_action = action
            
            # Track portfolio value (assumed to be directly proportional to cumulative reward)
            total_value += reward
            portfolio_values.append(total_value)
        
        rewards.append(total_reward)
        transaction_costs.append(total_transaction_cost)
        cumulative_returns.append(total_value - 1)  # Cumulative return over the episode
        
        # Calculate Sharpe and Sortino ratios for each episode
        mean_return = np.mean(cumulative_returns)
        std_return = np.std(cumulative_returns)
        sharpe_ratio = mean_return / std_return if std_return != 0 else 0
        
        downside_returns = [r for r in cumulative_returns if r < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = mean_return / downside_deviation if downside_deviation != 0 else 0
        
        episode_rewards.append(total_reward)
        episode_transaction_costs.append(total_transaction_cost)
        episode_cumulative_returns.append(total_value - 1)
        episode_sharpe_ratios.append(sharpe_ratio)
        episode_sortino_ratios.append(sortino_ratio)
        
        print(f"Episode {episode + 1}: Total Reward: {total_reward.item():.4f}, Total Transaction Cost: {total_transaction_cost.item():.4f}, Final Portfolio Value: {total_value.item():.4f}")


    
    avg_reward = np.mean(rewards)
    avg_transaction_cost = np.mean(transaction_costs)
    avg_cumulative_return = np.mean(cumulative_returns)
    
    # Sharpe ratio (mean return / standard deviation of returns)
    mean_return = np.mean(cumulative_returns)
    std_return = np.std(cumulative_returns)
    sharpe_ratio = mean_return / std_return if std_return != 0 else 0
    
    # Sortino ratio (mean return / downside deviation)
    downside_returns = [r for r in cumulative_returns if r < 0]
    downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
    sortino_ratio = mean_return / downside_deviation if downside_deviation != 0 else 0
    
    print(f"\nAverage Reward over {episodes} episodes: {avg_reward:.4f}")
    print(f"Average Transaction Cost over {episodes} episodes: {avg_transaction_cost:.4f}")
    print(f"Average Cumulative Return over {episodes} episodes: {avg_cumulative_return:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Sortino Ratio: {sortino_ratio:.4f}")
    
    # Visualization
    visualize_results(episode_rewards, episode_transaction_costs, episode_cumulative_returns, episode_sharpe_ratios, episode_sortino_ratios)

def visualize_results(rewards, transaction_costs, cumulative_returns, sharpe_ratios, sortino_ratios):
    """
    Visualizes the results of the evaluation: rewards, transaction costs, cumulative returns, and ratios.
    """
    # Set up the plotting style
    sns.set_theme(style="darkgrid")

    # Plot Cumulative Return
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns, label='Cumulative Return', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Return')
    plt.title('Cumulative Return per Episode')
    plt.legend()
    plt.show()

    # Plot Transaction Costs
    plt.figure(figsize=(10, 6))
    plt.plot(transaction_costs, label='Transaction Cost', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Transaction Cost')
    plt.title('Transaction Cost per Episode')
    plt.legend()
    plt.show()

    # Plot Rewards
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label='Rewards', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.title('Rewards per Episode')
    plt.legend()
    plt.show()

    # Plot Sharpe Ratio
    plt.figure(figsize=(10, 6))
    plt.plot(sharpe_ratios, label='Sharpe Ratio', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Sharpe Ratio')
    plt.title('Sharpe Ratio per Episode')
    plt.legend()
    plt.show()

    # Plot Sortino Ratio
    plt.figure(figsize=(10, 6))
    plt.plot(sortino_ratios, label='Sortino Ratio', color='purple')
    plt.xlabel('Episode')
    plt.ylabel('Sortino Ratio')
    plt.title('Sortino Ratio per Episode')
    plt.legend()
    plt.show()

# Run the evaluation
evaluate_model(model, env, episodes=25, transaction_cost=0.0001)

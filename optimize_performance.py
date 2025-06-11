from flask import Flask, jsonify, request
from stable_baselines3 import PPO
from flask_cors import CORS
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import PortfolioEnv
from dataloader import load_data
import numpy as np

app = Flask(__name__)
CORS(app)

# Load test data (use different data or split the original data into train/test)
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
log_returns = load_data(tickers, start="2023-04-10", end="2025-04-10")  # Test data

# Create environment
env = PortfolioEnv(log_returns, window_size=30, n_assets=5)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=False, norm_reward=True)

# Load the trained PPO model
model = PPO.load("ppo_portfolio_model_5n", env=env)

# Performance Optimization: Evaluating the model with enhanced performance metrics
def optimize_performance(model, env, episodes=25, transaction_cost=0.0001):
    rewards = []
    portfolio_values = []
    transaction_costs = []
    cumulative_returns = []
    
    # Data for output
    episode_rewards = []
    episode_transaction_costs = []
    episode_cumulative_returns = []
    episode_sharpe_ratios = []
    
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
        
        # Calculate Sharpe ratio
        mean_return = np.mean(cumulative_returns)
        std_return = np.std(cumulative_returns)
        sharpe_ratio = mean_return / std_return if std_return != 0 else 0
        
        episode_rewards.append(total_reward)
        episode_transaction_costs.append(total_transaction_cost)
        episode_cumulative_returns.append(total_value - 1)
        episode_sharpe_ratios.append(sharpe_ratio)
        
        print(f"Episode {episode + 1}: Total Reward: {total_reward.item():.4f}, Total Transaction Cost: {total_transaction_cost.item():.4f}, Final Portfolio Value: {total_value.item():.4f}")

    
    avg_reward = np.mean(rewards)
    avg_transaction_cost = np.mean(transaction_costs)
    avg_cumulative_return = np.mean(cumulative_returns)
    
    # Sharpe ratio (mean return / standard deviation of returns)
    mean_return = np.mean(cumulative_returns)
    std_return = np.std(cumulative_returns)
    sharpe_ratio = mean_return / std_return if std_return != 0 else 0
    
    print(f"\nAverage Reward over {episodes} episodes: {avg_reward:.4f}")
    print(f"Average Transaction Cost over {episodes} episodes: {avg_transaction_cost:.4f}")
    print(f"Average Cumulative Return over {episodes} episodes: {avg_cumulative_return:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    
    # Output the results as needed
    return {
        "avg_reward": round(float(avg_reward), 4),
        "avg_transaction_cost": round(float(avg_transaction_cost), 4),
        "avg_cumulative_return": round(float(avg_cumulative_return), 4),
        "sharpe_ratio": round(float(sharpe_ratio), 4)
    }

@app.route('/optimize-performance', methods=['POST'])
def optimize_performance_api():
    try:
        data = request.get_json()
        print(f"Received data: {data}")
        episodes = int(data.get('episodes', 25))
        transaction_cost = float(data.get('transaction_cost', 0.0001))
        tickers = data.get('companies', [])
        tickers = [company['ticker'] for company in tickers]
        n_assets = len(tickers)

        # Load data dynamically
        log_returns = load_data(tickers, start="2023-04-10", end="2025-04-10")
        env = PortfolioEnv(log_returns, window_size=30, n_assets=n_assets)
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=False, norm_reward=True)

        # Choose correct model
        model_path = {
            3: "ppo_portfolio_model_3n",
            5: "ppo_portfolio_model_5n",
            10: "ppo_portfolio_model"
        }.get(n_assets)

        if model_path is None:
            return jsonify({"error": f"Unsupported number of assets: {n_assets}"}), 400

        model = PPO.load(model_path, env=env)

        results = optimize_performance(model, env, episodes, transaction_cost)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)

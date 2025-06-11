from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
from env import PortfolioEnv
from dataloader import load_data
import numpy as np

# Load log returns
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "JPM", "UNH", "HD", "V"]
log_returns = load_data(tickers, start="2022-01-01", end="2025-01-01")

# Create vectorized env
env = DummyVecEnv([lambda: PortfolioEnv(log_returns, window_size=30)])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Load the previous model
model = RecurrentPPO.load("recurrent_ppo_portfolio_model", env=env)

# Define learning rate schedule (decrease LR over time)
def lr_schedule(progress):
    """Learning rate schedule: Linearly decreasing from 1e-4 to 1e-6."""
    return 1e-4 * (1 - progress) + 1e-6 * progress

# Apply the learning rate scheduler to the model
model.learning_rate = lr_schedule

# Custom evaluation function
def evaluate_model(model, env, eval_episodes=5):
    """Evaluate the model performance."""
    total_rewards = []
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)
    
    mean_reward = np.mean(total_rewards)
    return mean_reward

# Continue training and evaluate every 15 iterations
total_timesteps = 1000000  # You can set this to the number of steps you want to train further
eval_freq = 15

# Start training
for iteration in range(total_timesteps // 1024):
    model.learn(total_timesteps=1024, reset_num_timesteps=False)
    
    # Perform evaluation every 15 iterations
    if iteration % eval_freq == 0:
        print(f"\nEvaluating after iteration {iteration}...")
        mean_reward = evaluate_model(model, env)
        print(f"Mean reward after iteration {iteration}: {mean_reward}")
    
# Save the trained model after further training
model.save("recurrent_ppo_portfolio_model")
print("Training completed and model saved!")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from env import PortfolioEnv
from dataloader import load_data

# Load log returns
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "JPM", "UNH", "HD", "V"]
log_returns = load_data(tickers, start="2020-01-01", end="2025-01-01")

print(log_returns.shape)
print(log_returns.head())


# Create environment
env = Monitor(PortfolioEnv(log_returns, window_size=30))
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=False, norm_reward=True)

# Create a callback to save the best model
eval_callback = EvalCallback(
    env,
    best_model_save_path="./best_model_ppo",
    log_path="./logs",
    eval_freq=10000,
    deterministic=True,
    render=False,
)

# Set up PPO model
model = PPO(
    "MlpPolicy",  
    env,
    verbose=1,
    tensorboard_log="./tensorboard_logs", 
    learning_rate=1e-4,  
    batch_size=64,      
    n_steps=512,        
    gamma=0.99,          
    gae_lambda=0.95,     
    ent_coef=0.02,       
    clip_range=0.2,      
    n_epochs=5,          
    target_kl=0.05,      
)


# Train the model
model.learn(total_timesteps=300_000, callback=eval_callback)

# Save the model
model.save("ppo_portfolio_model")

print("Training completed and model saved!")

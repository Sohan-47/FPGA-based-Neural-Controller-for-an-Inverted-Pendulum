import gymnasium as gym
import numpy as np
import os
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import torch.nn as nn

# (Force -10N to 10N)
class SimpleContinuousCartPole(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    
    def step(self, action):
        force = float(action[0]) * 10.0
        
        # Physics Engine
        x, x_dot, theta, theta_dot = self.env.unwrapped.state
        gravity = 9.8
        masscart = 1.0
        masspole = 0.1
        total_mass = (masspole + masscart)
        length = 0.5 
        polemass_length = (masspole * length)
        tau = 0.02
        
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + polemass_length * theta_dot**2 * sintheta) / total_mass
        thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * costheta**2 / total_mass))
        xacc = temp - polemass_length * thetaacc * costheta / total_mass
        
        x = x + tau * x_dot
        x_dot = x_dot + tau * xacc
        theta = theta + tau * theta_dot
        theta_dot = theta_dot + tau * thetaacc
        
        self.env.unwrapped.state = (x, x_dot, theta, theta_dot)
        
        # Reward: +1 for every step alive. Max = 500.
        reward = 1.0
        
        terminated = bool(x < -2.4 or x > 2.4 or theta < -0.209 or theta > 0.209)
        return np.array(self.env.unwrapped.state, dtype=np.float32), reward, terminated, False, {}

# save best model
class SimpleLogCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]
            
            if "episode" in info:
                reward = info['episode']['r']
                self.episode_rewards.append(reward)
                
                # avg of 5 runs
                mean_reward = np.mean(self.episode_rewards[-5:])
                
                # save
                new_best_str = ""
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save("best_model") 
                    new_best_str = "(*NEW BEST*)"

                print(f"Run {len(self.episode_rewards):3d} | Balanced: {int(reward):3d} steps | Avg: {mean_reward:5.1f} | {new_best_str}")
        return True
    
env = SimpleContinuousCartPole(gym.make("CartPole-v1"))
env = gym.wrappers.RecordEpisodeStatistics(env)

# 8 neurons, ReLU 
policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=dict(pi=[8], vf=[8]))

print("\n--- TRAINING START (Max Score: 500) ---")
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=0, learning_rate=0.002)

model.learn(total_timesteps=50000, callback=SimpleLogCallback())


print("\n" + "="*50)
print("   LOADING BEST MODEL & EXPORTING")
print("="*50)


if os.path.exists("best_model.zip"):
    model = PPO.load("best_model")

# layer 1
w1 = (model.policy.mlp_extractor.policy_net[0].weight.detach().cpu().numpy() * 128).astype(int).flatten()
b1 = (model.policy.mlp_extractor.policy_net[0].bias.detach().cpu().numpy() * 128).astype(int).flatten()

print("\n// Layer 1 Weights (8 Neurons)")
for i, val in enumerate(w1):
    print(f"w1[{i}] = {val};", end=" ")
    if (i+1)%4==0: print("")

print("\n// Layer 1 Biases")
for i, val in enumerate(b1):
    print(f"b1[{i}] = {val};", end=" ")

# layer 2 
w2 = (model.policy.action_net.weight.detach().cpu().numpy() * 128).astype(int).flatten()
b2 = (model.policy.action_net.bias.detach().cpu().numpy() * 128).astype(int)

# verilog parameters :

print("\n\n// Layer 2 Weights")
for i, val in enumerate(w2):
    print(f"w2[{i}] = {val};", end=" ")

print(f"\n\n// Layer 2 Bias\nb2 = {b2[0]};")
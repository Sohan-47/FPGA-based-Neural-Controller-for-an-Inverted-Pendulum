import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(4, 8)  
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

env = gym.make("CartPole-v1")
model = SimpleNet()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

memory = deque(maxlen=2000)
epsilon = 1.0  
epsilon_min = 0.01
epsilon_decay = 0.995


best_reward = 0.0      
consecutive_wins = 0   

print("Initiating training.....")


for episode in range(1600): 
    state, _ = env.reset()
    state = torch.FloatTensor(state)
    total_reward = 0
    done = False
    
    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample() 
        else:
            with torch.no_grad():
                q_values = model(state)
                action = torch.argmax(q_values).item()

        next_state_np, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        next_state = torch.FloatTensor(next_state_np)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if len(memory) > 32:
            batch = random.sample(memory, 32)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            states = torch.stack(states)
            next_states = torch.stack(next_states)
            rewards = torch.tensor(rewards)
            dones = torch.tensor(dones, dtype=torch.float32)
            
            current_q = model(states).gather(1, torch.tensor(actions).unsqueeze(1)).squeeze()
            next_q = model(next_states).max(1)[0]
            target_q = rewards + (0.99 * next_q * (1 - dones))
            
            loss = loss_fn(current_q, target_q.detach())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
        

    if total_reward > best_reward:
        best_reward = total_reward
        torch.save(model.state_dict(), "simple_net_best.pth")
        print(f"Episode {episode}: New Record ({best_reward})! -> Model Saved.")
        
    
    if total_reward >= 500:
        consecutive_wins += 1
        print(f"Episode {episode}: Consistent Run Observed : ({consecutive_wins}/5)")
    else:
        consecutive_wins = 0 
        
    if consecutive_wins >= 5:
        print("\nSUCCESS: Model has solved the environment consistently.")
        print("Early stop activated.")
        break
        
    if episode % 50 == 0:
        print(f"Episode {episode}, Reward: {total_reward}, Best So Far: {best_reward}, Epsilon: {epsilon:.2f}")

print("Training Complete.")
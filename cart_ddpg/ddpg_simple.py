import torch
import torch.nn as nn
import numpy as np
import random
from models import Actor, Critic, ResBlock
import torch.optim as optim
from collections import deque 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, image, aim_point, action, reward, next_image, next_aim_point, done):
        self.buffer.append((image, aim_point, action, reward, next_image, next_aim_point, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        images, aim_points, actions, rewards, next_images, next_aim_points, dones = zip(*batch)
        images = torch.stack(images)
        aim_points = torch.stack(aim_points)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_images = torch.stack(next_images)
        next_aim_points = torch.stack(next_aim_points)
        dones = torch.stack(dones)
        return images, aim_points, actions, rewards, next_images, next_aim_points, dones

    def __len__(self):
        return len(self.buffer)

class DDPG:
    def __init__(self, buffer_size=10000, gamma=0.8, tau=0.005, lr_actor=1e-4, lr_critic=1e-3):
        self.actor = Actor(ResBlock).to(device)

        self.critic = Critic(ResBlock).to(device)
        self.target_actor = Actor(ResBlock).to(device)

        self.target_critic = Critic(ResBlock).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.gamma = gamma
        self.tau = tau

        # Initialize target networks with the same weights
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def select_action(self, image, aim_point, noise=0.2):
        image = image.unsqueeze(0).to(device)
        aim_point = aim_point.unsqueeze(0).to(device)

        action = self.actor(image, aim_point).cpu().detach().numpy()[0]
        print(action)
        action += (noise * np.random.randn(action.shape[0]))  # Add noise for exploration
        
        return action

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return 100, 100

        images, aim_points, actions, rewards, next_images, next_aim_points, dones = self.replay_buffer.sample(batch_size)
        
        images = images.to(device)
        aim_points = aim_points.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_images = next_images.to(device)
        next_aim_points = next_aim_points.to(device)
        dones = dones.to(device)

        # Update Critic
        with torch.no_grad():
            target_actions = self.target_actor(images, aim_points)
            target_q = self.target_critic(next_images, next_aim_points, target_actions)
            y = rewards.unsqueeze(1) + self.gamma * (1 - dones.unsqueeze(1)) * target_q

        q_values = self.critic(images, aim_points, actions)
        critic_loss = nn.MSELoss()(q_values, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        total_norm = 0
        for param in self.critic.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        # print(f"Total Gradient Norm: {total_norm}")

        # for name, param in self.critic.named_parameters():
        #     print(f"Layer {name}: Gradient Norm {param.grad.norm()}")
        self.critic_optimizer.step()

        # Update Actor
        actions = self.actor(images, aim_points)
        q_values = self.critic(images, aim_points, actions)
        actor_loss = (-q_values).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        total_norm = 0
        for param in self.actor.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(f"Total Gradient Norm: {total_norm}")

        # for name, param in self.actor.named_parameters():
        #     print(f"Layer {name}: Gradient Norm {param.grad.norm()}")

        self.actor_optimizer.step()

        # Update Target Networks
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        print(f"Actor Loss: {actor_loss}")
        print(f"Critic Loss: {critic_loss}")

        return actor_loss.cpu().detach().numpy(), critic_loss.cpu().detach().numpy()

    def store_transition(self, image, aim_point, action, reward, next_image, next_aim_point, done):
        self.replay_buffer.add(image, aim_point, action, reward, next_image, next_aim_point, done)
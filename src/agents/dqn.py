import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class DQNAgent:
    def __init__(self, input_dim, output_dim, lr=0.01):
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = []

    def act(self, state):
        if random.random() < 0.1:
            return random.randint(0, 1)
        with torch.no_grad():
            q_values = self.model(torch.FloatTensor(state))
            return int(torch.argmax(q_values).item())

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def learn(self):
        if len(self.memory) < 10:
            return
        batch = random.sample(self.memory, 10)
        for state, action, reward, next_state in batch:
            q_pred = self.model(torch.FloatTensor(state))[action]
            q_target = reward + 0.99 * torch.max(self.model(torch.FloatTensor(next_state)))
            loss = self.criterion(q_pred, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

import numpy as np
import gym
from gym import spaces

class LabelAggregationEnv(gym.Env):
    def __init__(self, num_agents=3, num_samples=100):
        super(LabelAggregationEnv, self).__init__()
        self.num_agents = num_agents
        self.num_samples = num_samples
        self.current_sample = 0

        self.labels = np.random.randint(0, 2, size=(num_samples,))
        self.agent_votes = np.zeros((num_agents, num_samples), dtype=int)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def reset(self):
        self.current_sample = 0
        return np.array([self.labels[self.current_sample]], dtype=np.float32)

    def step(self, actions):
        correct_label = self.labels[self.current_sample]
        rewards = [1 if action == correct_label else -1 for action in actions]

        self.current_sample += 1
        done = self.current_sample >= self.num_samples
        obs = np.array([self.labels[self.current_sample % self.num_samples]], dtype=np.float32)

        return [obs]*self.num_agents, rewards, done, {}

    def render(self, mode='human'):
        print(f"Sample {self.current_sample} Label: {self.labels[self.current_sample % self.num_samples]}")

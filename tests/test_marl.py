import unittest
from src.env.label_env import LabelAggregationEnv
from src.agents.dqn import DQNAgent

class TestEnv(unittest.TestCase):
    def test_env_step(self):
        env = LabelAggregationEnv(num_agents=3)
        obs = env.reset()
        actions = [0, 1, 1]
        next_obs, rewards, done, _ = env.step(actions)
        self.assertEqual(len(rewards), 3)
        self.assertFalse(done or next_obs is None)

class TestAgent(unittest.TestCase):
    def test_agent_act_learn(self):
        agent = DQNAgent(input_dim=1, output_dim=2)
        s, ns = [0.0], [1.0]
        a = agent.act(s)
        agent.remember(s, a, 1, ns)
        agent.learn()
        self.assertIn(a, [0, 1])

if __name__ == '__main__':
    unittest.main()

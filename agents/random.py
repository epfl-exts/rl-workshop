# Below are implementations of some standard RL agents
class RandomAgent():
    """Random agent"""

    def __init__(self, env):
        self.env = env

    def act(self, state):
        return self.env.action_space.sample()

    def reset(self):
        pass

    def learn(self, state, action, reward, next_state, done):
        pass

from agent_model.model import Model
from agent_model.training_agent import TrainingAgent
import numpy as np


class TestModel(Model):
    def model_logic(self, state, p):
        d = np.random.randint(low=0, high=4)
        # i0 = Math.Max(0, i1 - d);
        # i1 = Math.Max(0, i2 - Math.Max(0, d - i1));
        # i2 = Math.Max(0, p - Math.Max(0, Math.Max(0, d - i1) - i2));
        i0 = state[0]
        i1 = state[1]
        i2 = state[2]
        next_state = [max(0, i1 - d), max(0, i2 - max(0, d - i1)), max(0, p - max(0, max(0, d - i1) - i2))]

        ni0 = next_state[0]
        ni1 = next_state[1]
        ni2 = next_state[2]

        reward = 100 * (min(0, i0 + i1 + i2 + p - d)) - (0 * (ni0 + ni1 + ni2)) + 10 * (min(0, i0 - d))

        return state, p, next_state, reward, False


initial_state = [0, 0, 0]
model = TestModel(initial_state, 4, 3)
agent = TrainingAgent(model=model, runs=200, steps_per_run=365, batch_size=30, epsilon_decay=0.0002)
agent.run()

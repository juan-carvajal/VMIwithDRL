from agent_model.training_agent_torch2 import TrainingAgent
from implementation.VMImodel import VMI

initial_state = [0, 0, 0, 0, 0]
#print(tensorflow.test.is_gpu_available())
model = VMI(4, 100, 5, initial_state, 5, 100)
agent = TrainingAgent(model=model, runs=1000, steps_per_run=365, batch_size=500,memory=10000,use_gpu=True)

agent.run(validateRuns=10
           )

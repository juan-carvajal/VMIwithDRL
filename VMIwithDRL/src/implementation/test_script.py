'''
Created on 17/11/2019

@author: juan0
'''
from agent_model.training_agent_torch3 import TrainingAgent
from implementation.VMImodel import VMI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


initial_state = [0, 0, 0, 0, 0,1]
#print(tensorflow.test.is_gpu_available())
model = VMI(4, 100, 5, initial_state, 5, 100)
agent = TrainingAgent(model=model, runs=150, steps_per_run=365, batch_size=32,memory=5000,use_gpu=True,epsilon_function='linear',min_epsilon=0.01,epsilon_min_percentage=0.25)
rewards=agent.run()
log=model.log
log_df=pd.DataFrame(log)
print(log_df)

df=pd.DataFrame(rewards,columns=['rewards'])
df.reset_index(level=0, inplace=True)
df.columns=['index','data']
rolling_mean = df.data.rolling(window=50).mean()
plt.plot(df.index, df.data, label='Rewards')
plt.plot(df.index, rolling_mean, label='SMA(n=50)', color='orange')
plt.legend(loc='upper left')
plt.show()


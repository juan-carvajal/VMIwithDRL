'''
Created on 17/11/2019

@author: juan0
'''
from agent_model.training_agent_torch3 import TrainingAgent
from implementation.VMImodel import VMI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean 


initial_state = [0, 0, 0, 0, 0,1]
#print(tensorflow.test.is_gpu_available())
train_runs=250
model = VMI(4, 100, 5, initial_state, 5, 100)
agent = TrainingAgent(model=model, runs=train_runs, steps_per_run=365, batch_size=32,memory=5000,use_gpu=True,epsilon_function='log',min_epsilon=0.01,epsilon_min_percentage=0.1)
rewards=agent.run()
log=model.log
expirees=[]
stockouts=[]
dataExport=[]
for year in range(train_runs):
    stk=0
    exp=0
    for day in range(len(log[year])):
        stk+=mean(log[year][day]["stockouts"])
        exp+=mean(log[year][day]["expirees"])
        dataExport.append([year,day,log[year][day]["action"]]+log[year][day]["inventory"]+[log[year][day]["reward"]])
    stk=stk/float(len(log[year]))
    exp=stk/float(len(log[year]))
    stockouts.append(stk)
    expirees.append(exp)
    
log_export=pd.DataFrame(dataExport)
log_export.reset_index(level=0, inplace=True)
log_export.columns=['index','year','day','action','I0','I1','I2','I3','I4','reward']
log_export.to_csv('data.csv')
    
log_data={"stockouts":stockouts,"expirees":expirees}
#print(log_data)
log_df=pd.DataFrame(log_data)
log_df.reset_index(level=0, inplace=True)
#print(log_df)
log_df.columns=['index','stockouts','expirees']
plt.plot(log_df.index, log_df.stockouts,label='Avg. Stockouts')
plt.plot(log_df.index, log_df.expirees,label='Avg. Expirees',color='orange')
plt.legend(loc='upper left')
plt.show()



df=pd.DataFrame(rewards,columns=['rewards'])
df.reset_index(level=0, inplace=True)
df.columns=['index','data']
rolling_mean = df.data.rolling(window=50).mean()
plt.plot(df.index, df.data, label='Rewards')
plt.plot(df.index, rolling_mean, label='SMA(n=50)', color='orange')
plt.legend(loc='upper left')
plt.show()


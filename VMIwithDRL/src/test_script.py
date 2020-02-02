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

if __name__ == '__main__':
    initial_state = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    # print(tensorflow.test.is_gpu_available())
    train_runs = 50
    model = VMI(4, 100, 5, initial_state, 5, 100)
<<<<<<< HEAD
<<<<<<< HEAD
    agent = TrainingAgent(model=model, runs=train_runs, steps_per_run=365, batch_size=32, memory=1825, use_gpu=True,
                          epsilon_function='log', min_epsilon=0, epsilon_min_percentage=0.1)
=======
    agent = TrainingAgent(model=model, runs=train_runs, steps_per_run=365, batch_size=150, memory=5000, use_gpu=False,
                          epsilon_function='linear', min_epsilon=0.01, epsilon_min_percentage=0.3)
>>>>>>> ad0fc2e7fc01b79f0ed458e94c2ec1e7ddceedc3
=======
    agent = TrainingAgent(model=model, runs=train_runs, steps_per_run=365, batch_size=150, memory=5000, use_gpu=False,
                          epsilon_function='linear', min_epsilon=0.01, epsilon_min_percentage=0.3)
>>>>>>> ad0fc2e7fc01b79f0ed458e94c2ec1e7ddceedc3
    rewards = agent.run()
    log = model.log
    expirees = []
    stockouts = []
    dc_expirees = []
    dataExport = []
    opt_use = []
    for year in range(train_runs):
        stk = 0
        exp = 0
        dc_exp = 0
        opt = 0
        for day in range(len(log[year])):
            stk += sum(log[year][day]["stockouts"])
            exp += sum(log[year][day]["expirees"])
            dc_exp += log[year][day]["DC_expirees"]
            # all=[list(x) for x in zip(*log[year][day]["allocation"])]
            if log[year][day]['Used_LP_Model']:
                opt += 1

            all = log[year][day]["allocation"]
            a = [year, day, log[year][day]["action"]] + log[year][day]["inventory"] + [log[year][day]["reward"]] + \
                log[year][day]["demands"] + [log[year][day]["donors"]] + log[year][day]["stockouts"] + log[year][day][
                    "expirees"] + [log[year][day]["DC_expirees"]] + [item for sublist in all for item in sublist] + [
                    item
                    for
                    sublist
                    in log[
                        year][
                        day][
                        "II"]
                    for
                    item in
                    sublist]
            dataExport.append(a)

        opt_use.append(opt / len(log[year]))
        stockouts.append(stk)
        expirees.append(exp)
        dc_expirees.append(dc_exp)

    log_export = pd.DataFrame(dataExport)
    log_export.reset_index(level=0, inplace=True)
    log_export.columns = ['index', 'year', 'day', 'action', 'I0', 'I1', 'I2', 'I3', 'I4', 'reward', 'D1', 'D2', 'D3',
                          'D4',
                          'donors', 'S1', 'S2', 'S3', 'S4', 'E1', 'E2', 'E3', 'E4', 'DC_E', 'H1_A0', 'H1_A1', 'H1_A2',
                          'H1_A3', 'H1_A4', 'H2_A0', 'H2_A1', 'H2_A2', 'H2_A3', 'H2_A4', 'H3_A0', 'H3_A1', 'H3_A2',
                          'H3_A3',
                          'H3_A4', 'H4_A0', 'H4_A1', 'H4_A2', 'H4_A3', 'H4_A4', 'H1_II0', 'H1_II1', 'H1_II2', 'H1_II3',
                          'H1_II4', 'H2_II0', 'H2_II1', 'H2_II2', 'H2_II3', 'H2_II4', 'H3_II0', 'H3_II1', 'H3_II2',
                          'H3_II3', 'H3_II4', 'H4_II0', 'H4_II1', 'H4_II2', 'H4_II3', 'H4_II4']
    log_export.to_csv('data.csv')

    log_data = {"stockouts": stockouts, "expirees": expirees, "dc_expirees": dc_expirees}
    # print(log_data)
    log_df = pd.DataFrame(log_data)
    log_df.reset_index(level=0, inplace=True)
    # print(log_df)
    log_df.columns = ['index', 'stockouts', 'expirees', 'dc_expirees']
    plt.plot(log_df.index, log_df.stockouts, label='Stockouts')
    plt.plot(log_df.index, log_df.expirees, label='Expirees', color='orange')
    plt.plot(log_df.index, log_df.dc_expirees, label='DC Expirees', color='green')
    plt.legend(loc='upper left')
    plt.show()

    df = pd.DataFrame(rewards, columns=['rewards'])
    df.reset_index(level=0, inplace=True)
    df.columns = ['index', 'data']
    rolling_mean = df.data.rolling(window=50).mean()
    plt.plot(df.index, df.data, label='Rewards')
    plt.plot(df.index, rolling_mean, label='SMA(n=50)', color='orange')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

    opt_df = pd.DataFrame(opt_use, columns=['opt'])
    opt_df.reset_index(level=0, inplace=True)
    opt_df.columns = ['index', 'opt']
    plt.plot(opt_df.index, opt_df.opt, label='Opt. Model Use')
    plt.legend(loc='upper left')
    plt.show()

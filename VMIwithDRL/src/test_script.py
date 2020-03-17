'''
Created on 17/11/2019
@author: juan0
'''
# from agent_model.training_agent_torch3 import TrainingAgent
# from agent_model.DDQL_agent import TrainingAgent
from agent_model.ClippedDDQN_agent import TrainingAgent
# from implementation.VMImodel import VMI
from implementation.VMImodelVariable import VMI
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage, AnnotationBbox)
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import matplotlib.pyplot as plt
import os
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from statistics import mean,stdev
import numpy as np
import seaborn as sns
import pandas as pd


def send_mail():
    with open('recipients.txt', 'r') as file:
        recipients = file.read().splitlines()
    for recipient in recipients:

        msg = MIMEMultipart()
        msg['Subject'] = 'Training Report'
        msg['From'] = 'juancarvajal3@pepisandbox.com'
        msg['To'] = recipient
        images = ['output/train_reward.png', 'output/train_politic.png', 'output/stockouts_politic.png',
                  'output/expirees_politic.png', 'output/dc_expirees_politic.png',
                  'output/validate_reward.png', "output/production_level.png", "output/shipment_politic.png"]
        for image in images:
            img_data = open(image, 'rb').read()
            image_mime = MIMEImage(img_data, name=os.path.basename(image))
            msg.attach(image_mime)

        s = smtplib.SMTP("smtp.pepipost.com", 587)
        s.ehlo()
        s.starttls()
        s.ehlo()
        s.login('juancarvajal3', 'juancarvajal3_f00b649eb8bf91463d8dc5708b2c59cc')
        s.sendmail('juancarvajal3@pepisandbox.com', recipient, msg.as_string())
        s.quit()


if __name__ == '__main__':
    hosp_inv = [0] * 20
    initial_state = [10, 10, 10, 10, 10, 1] + hosp_inv
    # print(tensorflow.test.is_gpu_available())
    # magic numbers: runs 150 , eppercentage:0.25 , min_ep:0.05 , batch:32 , memory:1825

    # Estrategia buena para realizar el entrenmiento
    # agent = TrainingAgent(model=model, runs=train_runs, steps_per_run=365, batch_size=128, memory=1825, use_gpu=True,
    #                       epsilon_function='linear', min_epsilon=0.001, epsilon_min_percentage=0.25, lr=0.0005)

    # agent = TrainingAgent(model=model, runs=train_runs, steps_per_run=365, batch_size=32, memory=10000, use_gpu=True,
    #                       epsilon_function='linear', min_epsilon=0.005, epsilon_min_percentage=0.2)
    train_runs = 20
    model = VMI(4, 100, 5, train_runs, initial_state, 5, 100)
    agent = TrainingAgent(model=model, runs=train_runs, steps_per_run=365, batch_size=10, memory=127750, use_gpu=True,
                          epsilon_function='logv2', min_epsilon=0.01, epsilon_min_percentage=0.1)
    rewards = agent.run()
    validate_runs = 500
    val_rewards = agent.validate(validate_runs, 365)
    print("Total state space: ", [len(x) for x in model.state_space_memory])
    log = model.log["train"]
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
            a = [year, day, log[year][day]["shipment_size"]] + log[year][day]["inventory"] + [
                log[year][day]["reward"]] + \
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
                    sublist] + [log[year][day]["production_level"]]
            dataExport.append(a)
        opt_use.append(opt / len(log[year]))
        stockouts.append(stk)
        expirees.append(exp)
        dc_expirees.append(dc_exp)

    log_export = pd.DataFrame(dataExport)
    log_export.reset_index(level=0, inplace=True)
    log_export.columns = ['index', 'year', 'day', 'shipment_size', 'I0', 'I1', 'I2', 'I3', 'I4', 'reward', 'D1', 'D2',
                          'D3',
                          'D4',
                          'donors', 'S1', 'S2', 'S3', 'S4', 'E1', 'E2', 'E3', 'E4', 'DC_E', 'H1_A0', 'H1_A1', 'H1_A2',
                          'H1_A3', 'H1_A4', 'H2_A0', 'H2_A1', 'H2_A2', 'H2_A3', 'H2_A4', 'H3_A0', 'H3_A1', 'H3_A2',
                          'H3_A3',
                          'H3_A4', 'H4_A0', 'H4_A1', 'H4_A2', 'H4_A3', 'H4_A4', 'H1_II0', 'H1_II1', 'H1_II2', 'H1_II3',
                          'H1_II4', 'H2_II0', 'H2_II1', 'H2_II2', 'H2_II3', 'H2_II4', 'H3_II0', 'H3_II1', 'H3_II2',
                          'H3_II3', 'H3_II4', 'H4_II0', 'H4_II1', 'H4_II2', 'H4_II3', 'H4_II4', 'production_level']
    log_export.to_csv('output/train.csv')

    log_data = {"stockouts": stockouts, "expirees": expirees, "dc_expirees": dc_expirees}
    # print(log_data)
    log_df = pd.DataFrame(log_data)
    log_df.reset_index(level=0, inplace=True)
    # print(log_df)
    line_tc = 0.8
    log_df.columns = ['index', 'stockouts', 'expirees', 'dc_expirees']
    fig, ax = plt.subplots()
    ax.plot(log_df.index, log_df.stockouts, label='Stockouts', linewidth=line_tc)
    ax.plot(log_df.index, log_df.expirees, label='Expirees', linewidth=line_tc)
    ax.plot(log_df.index, log_df.dc_expirees, label='DC Expirees', linewidth=line_tc)
    plt.ylabel("Accumulated over episode")
    plt.xlabel("Episode")
    plt.title("Policy over episodes (Training)")
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('output/train_politic.png', dpi=300)
    plt.savefig('output/train_politic.svg', dpi=300)
    plt.show()
    plt.clf()

    df = pd.DataFrame(rewards, columns=['rewards'])
    df.reset_index(level=0, inplace=True)
    df.columns = ['index', 'data']
    rolling_mean = df.data.rolling(window=50).mean()
    fig, ax = plt.subplots()
    ax.plot(df.index, df.data, label='Rewards', linewidth=line_tc)
    ax.plot(df.index, rolling_mean, label='SMA(n=50)', linewidth=line_tc)
    xy = (len(rewards) - 1, rewards[-1])
    offsetbox = TextArea(xy[1], minimumdescent=False)
    ab = AnnotationBbox(offsetbox, xy,
                        xybox=(-10, 20),
                        xycoords='data',
                        boxcoords="offset points",
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)
    plt.legend(loc='best')
    plt.ylabel("Reward (Accumulated)")
    plt.xlabel("Episode")
    plt.title("Reward over episodes (Training)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('output/train_reward.png', dpi=300)
    plt.savefig('output/train_reward.svg', dpi=300)
    plt.show()
    plt.clf()

    log = model.log["validate"]
    expirees = []
    stockouts = []
    dc_expirees = []
    dataExport = []
    opt_use = []
    for year in log:
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
            a = [year, day, log[year][day]["shipment_size"]] + log[year][day]["inventory"] + [
                log[year][day]["reward"]] + \
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
                    sublist] + [log[year][day]["production_level"]]
            dataExport.append(a)
        opt_use.append(opt / len(log[year]))
        stockouts.append(stk)
        expirees.append(exp)
        dc_expirees.append(dc_exp)

    log_export = pd.DataFrame(dataExport)
    log_export.reset_index(level=0, inplace=True)
    log_export.columns = ['index', 'year', 'day', 'shipment_size', 'I0', 'I1', 'I2', 'I3', 'I4', 'reward', 'D1', 'D2',
                          'D3',
                          'D4',
                          'donors', 'S1', 'S2', 'S3', 'S4', 'E1', 'E2', 'E3', 'E4', 'DC_E', 'H1_A0', 'H1_A1', 'H1_A2',
                          'H1_A3', 'H1_A4', 'H2_A0', 'H2_A1', 'H2_A2', 'H2_A3', 'H2_A4', 'H3_A0', 'H3_A1', 'H3_A2',
                          'H3_A3',
                          'H3_A4', 'H4_A0', 'H4_A1', 'H4_A2', 'H4_A3', 'H4_A4', 'H1_II0', 'H1_II1', 'H1_II2', 'H1_II3',
                          'H1_II4', 'H2_II0', 'H2_II1', 'H2_II2', 'H2_II3', 'H2_II4', 'H3_II0', 'H3_II1', 'H3_II2',
                          'H3_II3', 'H3_II4', 'H4_II0', 'H4_II1', 'H4_II2', 'H4_II3', 'H4_II4', 'production_level']
    log_export.to_csv('output/evaluate.csv')

    log_data = {"stockouts": stockouts, "expirees": expirees, "dc_expirees": dc_expirees}
    # print(log_data)
    log_df = pd.DataFrame(log_data)
    log_df.reset_index(level=0, inplace=True)
    # print(log_df)
    parameters_x=0.01
    miu_y=0.04
    sigma_y=0.01
    e_data = log_data['stockouts']
    plt.grid(axis='y', alpha=0.5, linestyle='--', zorder=0)
    plt.hist(e_data, weights=np.zeros_like(e_data) + 1. / len(e_data), edgecolor='black', linewidth=1.2, zorder=100,color='C0')
    plt.xlabel('Total Stockouts')
    plt.ylabel('Frequency (Relative)')
    plt.title('Stockouts distribution plot (policy evaluation)')
    plt.axvline(mean(e_data), color='k', linestyle='dashed', linewidth=1, zorder=1000)
    plt.figtext(parameters_x, miu_y, r'$\mu={:.2f}$'.format(mean(e_data)))
    plt.figtext(parameters_x, sigma_y, r'$\sigma={:.2f}$'.format(stdev(e_data)))
    plt.tight_layout()
    plt.savefig('output/stockouts_politic.png', dpi=300)
    plt.savefig('output/stockouts_politic.svg', dpi=300)
    plt.show()
    plt.clf()

    e_data = log_data['expirees']
    plt.grid(axis='y', alpha=0.5, linestyle='--', zorder=0)
    plt.hist(e_data, weights=np.zeros_like(e_data) + 1. / len(e_data), edgecolor='black', linewidth=1.2, zorder=100,color='C1')
    plt.xlabel('Total Expirees')
    plt.ylabel('Frequency (Relative)')
    plt.title('Expirees distribution plot (policy evaluation)')
    plt.axvline(mean(e_data), color='k', linestyle='dashed', linewidth=1, zorder=1000)
    plt.figtext(parameters_x, miu_y, r'$\mu={:.2f}$'.format(mean(e_data)))
    plt.figtext(parameters_x, sigma_y, r'$\sigma={:.2f}$'.format(stdev(e_data)))
    plt.tight_layout()
    plt.savefig('output/expirees_politic.png', dpi=300)
    plt.savefig('output/expirees_politic.svg', dpi=300)
    plt.show()
    plt.clf()

    e_data = log_data['dc_expirees']
    plt.grid(axis='y', alpha=0.5, linestyle='--', zorder=0)
    plt.hist(e_data, weights=np.zeros_like(e_data) + 1. / len(e_data), edgecolor='black', linewidth=1.2, zorder=100,color='C2')
    plt.xlabel('Total DC Expirees')
    plt.ylabel('Frequency (Relative)')
    plt.title('DC expirees distribution plot (policy evaluation)')
    plt.axvline(mean(e_data), color='k', linestyle='dashed', linewidth=1, zorder=1000)
    plt.figtext(parameters_x, miu_y, r'$\mu={:.2f}$'.format(mean(e_data)))
    plt.figtext(parameters_x, sigma_y, r'$\sigma={:.2f}$'.format(stdev(e_data)))
    plt.tight_layout()
    plt.savefig('output/dc_expirees_politic.png', dpi=300)
    plt.savefig('output/dc_expirees_politic.svg', dpi=300)
    plt.show()
    plt.clf()


    # line_tc = 0.8
    # log_df.columns = ['index', 'stockouts', 'expirees', 'dc_expirees']
    # fig, ax = plt.subplots()
    # ax.plot(log_df.index, log_df.stockouts, label='Stockouts', linewidth=line_tc)
    # ax.plot(log_df.index, log_df.expirees, label='Expirees', linewidth=line_tc)
    # ax.plot(log_df.index, log_df.dc_expirees, label='DC Expirees', linewidth=line_tc)
    # plt.ylabel("Accumulated over episode")
    # plt.xlabel("Episode")
    # plt.title("Politic in validation")
    # plt.grid(True)
    # plt.legend(loc='best')
    # plt.tight_layout()
    # plt.savefig('output/validate_politic.png', dpi=300)
    # plt.savefig('output/validate_politic.svg', dpi=300)
    # plt.show()

    # df = pd.DataFrame(val_rewards, columns=['rewards'])
    # df.reset_index(level=0, inplace=True)
    # df.columns = ['index', 'data']
    # rolling_mean = df.data.rolling(window=50).mean()
    # fig, ax = plt.subplots()
    # ax.plot(df.index, df.data, label='Rewards', linewidth=line_tc)
    # m = mean(val_rewards)
    # ax.plot([0, len(val_rewards) - 1], [m, m], label="Mean", linewidth=line_tc, ls='dashed')
    # plt.legend(loc='best')
    # plt.ylabel("Reward (Accumulated)")
    # plt.xlabel("Episode")
    # plt.title("Reward in validation")
    # plt.grid(True)
    # plt.tight_layout()
    plt.grid(axis='y', alpha=0.5, linestyle='--', zorder=0)
    plt.hist(val_rewards, weights=np.zeros_like(val_rewards) + 1. / len(val_rewards), edgecolor='black', linewidth=1.2, zorder=100)
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency (Relative)')
    plt.title('Reward distribution plot (policy evaluation)')
    plt.axvline(mean(val_rewards), color='k', linestyle='dashed', linewidth=1, zorder=1000)
    min_ylim, max_ylim = plt.ylim()
    min_xlim, max_xlim = plt.xlim()
    plt.figtext(parameters_x, miu_y, r'$\mu={:.2f}$'.format(mean(val_rewards)))
    plt.figtext(parameters_x, sigma_y, r'$\sigma={:.2f}$'.format(stdev(val_rewards)))
    plt.tight_layout()
    plt.savefig('output/validate_reward.png', dpi=300)
    plt.savefig('output/validate_reward.svg', dpi=300)
    plt.show()
    plt.clf()

    A_data = []
    P_data = []

    for year in log:
        for day in log[year]:
            A_data.append((np.sum(day['II']), np.sum(day['inventory']), np.sum(day['shipment_size'])))
            P_data.append((np.sum(day['II']), np.sum(day['inventory']), np.sum(day['production_level'])))

    A_data = [t for t in (set(tuple(i) for i in A_data))]
    P_data = [t for t in (set(tuple(i) for i in P_data))]
    H_I, CD_I, A = zip(*A_data)

    # colors=[(1,0,0),(1,1,0),(0,1,0)]
    colors = [(0.95, 0.95, 0.95), (0, 0, 0)]
    cust_cmap = LinearSegmentedColormap.from_list("binary_blue", colors, N=100)

    from mpl_toolkits import mplot3d

    df = pd.DataFrame({'Hospital Inventory Position': H_I,
                       'CD Inventory position': CD_I,
                       'Shipment Size Politic Evaluation': A
                       })
    pivot = df.pivot_table(index='Hospital Inventory Position', columns='CD Inventory position',
                           values='Shipment Size Politic Evaluation')
    ax = sns.heatmap(pivot, cmap='viridis')
    ax.invert_yaxis()
    ax.set_title("Shipment Size Policy Evaluation")
    plt.tight_layout()
    plt.savefig('output/shipment_politic.png', dpi=300)
    plt.savefig('output/shipment_politic.svg', dpi=300)
    plt.show()
    plt.clf()

    # M = np.zeros((int(max(H_I) + 1), int(max(CD_I) + 1)))
    # M[H_I, CD_I] = A
    # ax = sns.heatmap(M, cmap=cust_cmap)
    # ax.invert_yaxis()
    # ax.set_xlabel('Hospital Inventory Position')
    # ax.set_ylabel('CD Inventory position')
    # ax.set_title("Shipment Size Politic Evaluation")
    # plt.tight_layout()
    # plt.savefig("output/shipment_politic.png", dpi=500)
    # plt.show()

    H_I, CD_I, P = zip(*P_data)

    df = pd.DataFrame({'Hospital Inventory Position': H_I,
                       'CD Inventory position': CD_I,
                       'Production Level Politic Evaluation': P
                       })
    pivot = df.pivot_table(index='Hospital Inventory Position', columns='CD Inventory position',
                           values='Production Level Politic Evaluation')
    ax = sns.heatmap(pivot, cmap='viridis')
    ax.invert_yaxis()
    ax.set_title("Production Level Policy Evaluation")
    plt.tight_layout()
    plt.savefig('output/production_level.png', dpi=300)
    plt.savefig('output/production_level.svg', dpi=300)
    plt.show()
    plt.clf()

    # M = np.zeros((max(H_I) + 1, max(CD_I) + 1))
    # M[H_I, CD_I] = P
    # ax = sns.heatmap(M, cmap=cust_cmap)
    # ax.invert_yaxis()
    # ax.set_xlabel('Hospital Inventory Position')
    # ax.set_ylabel('CD Inventory position')
    # ax.set_title("Production Level Politic Evaluation")
    # plt.tight_layout()
    # plt.savefig("output/production_level.png", dpi=500)
    # plt.show()

    send_mail()

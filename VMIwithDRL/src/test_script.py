'''
Created on 17/11/2019
@author: juan0
'''
#from agent_model.training_agent_torch3 import TrainingAgent
from agent_model.DDQL_agent import TrainingAgent
# from implementation.VMImodel import VMI
from implementation.VMImodelVariable import VMI
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage, AnnotationBbox)
import pandas as pd
import matplotlib.pyplot as plt
import os
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart


def send_mail():
    with open('recipients.txt', 'r') as file:
        recipients = file.read().splitlines()
    for recipient in recipients:

        msg = MIMEMultipart()
        msg['Subject'] = 'Training Report'
        msg['From'] = 'juancarvajal3@pepisandbox.com'
        msg['To'] = recipient
        images = ['output/train_reward.png', 'output/train_politic.png', 'output/model_use.png', 'output/q.png']
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
    initial_state = [10, 10, 10, 10, 10, 1, 0, 0, 0, 0]
    # print(tensorflow.test.is_gpu_available())
    # magic numbers: runs 150 , eppercentage:0.25 , min_ep:0.05 , batch:32 , memory:1825

    #Estrategia buena para realizar el entrenmiento cada año.
    # agent = TrainingAgent(model=model, runs=train_runs, steps_per_run=365, batch_size=365, memory=1825, use_gpu=True,
    #                       epsilon_function='linear', min_epsilon=0.01, epsilon_min_percentage=0.15)

    train_runs = 1000
    model = VMI(4, 100, 5,train_runs, initial_state, 5, 100)
    agent = TrainingAgent(model=model, runs=train_runs, steps_per_run=365, batch_size=32, memory=1825, use_gpu=True,
                          epsilon_function='linear', min_epsilon=0.01,epsilon_min_percentage=0.10)
    rewards = agent.run()
    agent.validate(50, 365)
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
            a = [year, day, log[year][day]["shipment_size"]] + log[year][day]["inventory"] + [log[year][day]["reward"]] + \
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
                    sublist]+[log[year][day]["production_level"]]
            dataExport.append(a)
        opt_use.append(opt / len(log[year]))
        stockouts.append(stk)
        expirees.append(exp)
        dc_expirees.append(dc_exp)

    log_export = pd.DataFrame(dataExport)
    log_export.reset_index(level=0, inplace=True)
    log_export.columns = ['index', 'year', 'day', 'shipment_size', 'I0', 'I1', 'I2', 'I3', 'I4', 'reward', 'D1', 'D2', 'D3',
                          'D4',
                          'donors', 'S1', 'S2', 'S3', 'S4', 'E1', 'E2', 'E3', 'E4', 'DC_E', 'H1_A0', 'H1_A1', 'H1_A2',
                          'H1_A3', 'H1_A4', 'H2_A0', 'H2_A1', 'H2_A2', 'H2_A3', 'H2_A4', 'H3_A0', 'H3_A1', 'H3_A2',
                          'H3_A3',
                          'H3_A4', 'H4_A0', 'H4_A1', 'H4_A2', 'H4_A3', 'H4_A4', 'H1_II0', 'H1_II1', 'H1_II2', 'H1_II3',
                          'H1_II4', 'H2_II0', 'H2_II1', 'H2_II2', 'H2_II3', 'H2_II4', 'H3_II0', 'H3_II1', 'H3_II2',
                          'H3_II3', 'H3_II4', 'H4_II0', 'H4_II1', 'H4_II2', 'H4_II3', 'H4_II4','production_level']
    log_export.to_csv('output/data.csv')

    log_data = {"stockouts": stockouts, "expirees": expirees, "dc_expirees": dc_expirees}
    # print(log_data)
    log_df = pd.DataFrame(log_data)
    log_df.reset_index(level=0, inplace=True)
    # print(log_df)
    line_tc = 0.7
    log_df.columns = ['index', 'stockouts', 'expirees', 'dc_expirees']
    fig, ax = plt.subplots()
    ax.plot(log_df.index, log_df.stockouts, label='Stockouts', linewidth=line_tc)
    ax.plot(log_df.index, log_df.expirees, label='Expirees', linewidth=line_tc)
    ax.plot(log_df.index, log_df.dc_expirees, label='DC Expirees', linewidth=line_tc)
    plt.ylabel("Accumulated over episode")
    plt.xlabel("Episode")
    plt.title("Politic over time")
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.savefig('output/train_politic.png', dpi=300)
    plt.show()

    df = pd.DataFrame(rewards, columns=['rewards'])
    df.reset_index(level=0, inplace=True)
    df.columns = ['index', 'data']
    rolling_mean = df.data.rolling(window=50).mean()
    fig, ax = plt.subplots()
    ax.plot(df.index, df.data, label='Rewards', linewidth=line_tc)
    ax.plot(df.index, rolling_mean, label='SMA(n=50)', color='orange', linewidth=line_tc)
    xy = (len(rewards) - 1, rewards[-1])
    offsetbox = TextArea(xy[1], minimumdescent=False)
    ab = AnnotationBbox(offsetbox, xy,
                        xybox=(-10, 20),
                        xycoords='data',
                        boxcoords="offset points",
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)
    plt.legend(loc='upper left')
    plt.ylabel("Reward (Accumulated)")
    plt.xlabel("Episode")
    plt.title("Reward over time")
    plt.grid(True)
    plt.savefig('output/train_reward.png', dpi=300)
    plt.show()

    # opt_df = pd.DataFrame(opt_use, columns=['opt'])
    # opt_df.reset_index(level=0, inplace=True)
    # opt_df.columns = ['index', 'opt']
    # plt.plot(opt_df.index, opt_df.opt, label='Opt. Model Use',linewidth=line_tc)
    # plt.legend(loc='upper left')
    # plt.savefig('output/model_use.png', dpi=300)
    # plt.show()
    send_mail()

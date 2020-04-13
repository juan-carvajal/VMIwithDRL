import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import TextArea, AnnotationBbox
import numpy as np
import seaborn as sns
from statistics import mean,stdev
from matplotlib import rc


if __name__ == '__main__':
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    rc('text', usetex=True)
    log_df = pd.read_csv("output/train.csv")
    log_df = log_df.groupby('year').sum()
    log_df['reward']=log_df['reward']*-1
    log_df['stockouts'] = log_df["S1"] + log_df["S2"] + log_df["S3"] + log_df["S4"]
    log_df['expirees'] = log_df["E1"] + log_df["E2"] + log_df["E3"] + log_df["E4"]
    log_df['year']=log_df.index
    line_tc = 0.8
    fig, ax = plt.subplots()
    ax.plot(log_df.year, log_df.stockouts, label='Stockouts', linewidth=line_tc)
    ax.plot(log_df.year, log_df.expirees, label='Expirees', linewidth=line_tc)
    ax.plot(log_df.year, log_df.DC_E, label='DC Expirees', linewidth=line_tc)
    plt.axvline(350, color='red', linestyle='dashed', linewidth=1, zorder=1000)
    plt.text(370,plt.ylim()[1]-750,"Training Start",color="red")
    plt.ylabel("Accumulated over episode")
    plt.xlabel("Episode")
    plt.title("Policy over episodes (Training)")
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('output_post/train_politic.png', dpi=300)
    plt.savefig('output_post/train_politic.svg', dpi=300)
    plt.show()
    plt.clf()
    rewards=log_df.reward.tolist()
    df = pd.DataFrame(rewards, columns=['rewards'])
    df.reset_index(level=0, inplace=True)
    df.columns = ['index', 'data']
    rolling_mean = df.data.rolling(window=50).mean()
    fig, ax = plt.subplots()
    ax.plot(df.index, df.data, label='Rewards', linewidth=line_tc)
    ax.plot(df.index, rolling_mean, label='SMA(n=50)', linewidth=line_tc)
    plt.axvline(350, color='red', linestyle='dashed', linewidth=1, zorder=1000)
    plt.text(370,plt.ylim()[0]+7000,"Training Start",color="red")
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
    plt.savefig('output_post/train_reward.png', dpi=300)
    plt.savefig('output_post/train_reward.svg', dpi=300)
    plt.show()
    plt.clf()

    eval_df=pd.read_csv("output/evaluate.csv")
    log_df = eval_df.groupby("year").sum()
    log_df['reward']=log_df['reward']*-1
    log_df['stockouts'] = log_df["S1"] + log_df["S2"] + log_df["S3"] + log_df["S4"]
    log_df['expirees'] = log_df["E1"] + log_df["E2"] + log_df["E3"] + log_df["E4"]
    #log_df['year']=log_df.index
    log_df.reset_index(level=0, inplace=True)
    parameters_x = 0.01
    miu_y = 0.04
    sigma_y = 0.01
    e_data = log_df['stockouts']
    plt.grid(axis='y', alpha=0.5, linestyle='--', zorder=0)
    plt.hist(e_data, weights=np.zeros_like(e_data) + 1. / len(e_data), edgecolor='black', linewidth=1.2, zorder=100,
             color='C0')
    plt.xlabel('Total Stockouts')
    plt.ylabel('Frequency (Relative)')
    plt.title('Stockouts distribution plot (policy evaluation)')
    plt.axvline(mean(e_data), color='k', linestyle='dashed', linewidth=1, zorder=1000)
    plt.figtext(parameters_x, miu_y, r'$\mu={:.2f}$'.format(mean(e_data)))
    plt.figtext(parameters_x, sigma_y, r'$\sigma={:.2f}$'.format(stdev(e_data)))
    plt.tight_layout()
    plt.savefig('output_post/stockouts_politic.png', dpi=300)
    plt.savefig('output_post/stockouts_politic.svg', dpi=300)
    plt.show()
    plt.clf()

    e_data = log_df['expirees']
    plt.grid(axis='y', alpha=0.5, linestyle='--', zorder=0)
    plt.hist(e_data, weights=np.zeros_like(e_data) + 1. / len(e_data), edgecolor='black', linewidth=1.2, zorder=100,
             color='C1')
    plt.xlabel('Total Expirees')
    plt.ylabel('Frequency (Relative)')
    plt.title('Expirees distribution plot (policy evaluation)')
    plt.axvline(mean(e_data), color='k', linestyle='dashed', linewidth=1, zorder=1000)
    plt.figtext(parameters_x, miu_y, r'$\mu={:.2f}$'.format(mean(e_data)))
    plt.figtext(parameters_x, sigma_y, r'$\sigma={:.2f}$'.format(stdev(e_data)))
    plt.tight_layout()
    plt.savefig('output_post/expirees_politic.png', dpi=300)
    plt.savefig('output_post/expirees_politic.svg', dpi=300)
    plt.show()
    plt.clf()

    e_data = log_df['DC_E']
    plt.grid(axis='y', alpha=0.5, linestyle='--', zorder=0)
    plt.hist(e_data, weights=np.zeros_like(e_data) + 1. / len(e_data), edgecolor='black', linewidth=1.2, zorder=100,
             color='C2')
    plt.xlabel('Total DC Expirees')
    plt.ylabel('Frequency (Relative)')
    plt.title('DC expirees distribution plot (policy evaluation)')
    plt.axvline(mean(e_data), color='k', linestyle='dashed', linewidth=1, zorder=1000)
    plt.figtext(parameters_x, miu_y, r'$\mu={:.2f}$'.format(mean(e_data)))
    plt.figtext(parameters_x, sigma_y, r'$\sigma={:.2f}$'.format(stdev(e_data)))
    plt.tight_layout()
    plt.savefig('output_post/dc_expirees_politic.png', dpi=300)
    plt.savefig('output_post/dc_expirees_politic.svg', dpi=300)
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
    val_rewards=log_df['reward'].tolist()
    plt.grid(axis='y', alpha=0.5, linestyle='--', zorder=0)
    plt.hist(val_rewards, weights=np.zeros_like(val_rewards) + 1. / len(val_rewards), edgecolor='black', linewidth=1.2,
             zorder=100)
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency (Relative)')
    plt.title('Reward distribution plot (policy evaluation)')
    plt.axvline(mean(val_rewards), color='k', linestyle='dashed', linewidth=1, zorder=1000)
    min_ylim, max_ylim = plt.ylim()
    min_xlim, max_xlim = plt.xlim()
    plt.figtext(parameters_x, miu_y, r'$\mu={:.2f}$'.format(mean(val_rewards)))
    plt.figtext(parameters_x, sigma_y, r'$\sigma={:.2f}$'.format(stdev(val_rewards)))
    plt.tight_layout()
    plt.savefig('output_post/validate_reward.png', dpi=300)
    plt.savefig('output_post/validate_reward.svg', dpi=300)
    plt.show()
    plt.clf()

    # A_data = []
    # P_data = []
    #
    # for year in log:
    #     for day in log[year]:
    #         A_data.append((np.sum(day['II']), np.sum(day['inventory']), np.sum(day['shipment_size'])))
    #         P_data.append((np.sum(day['II']), np.sum(day['inventory']), np.sum(day['production_level'])))
    #
    # A_data = [t for t in (set(tuple(i) for i in A_data))]
    # P_data = [t for t in (set(tuple(i) for i in P_data))]
    # H_I, CD_I, A = zip(*A_data)
    #
    # # colors=[(1,0,0),(1,1,0),(0,1,0)]
    colors = [(0.95, 0.95, 0.95), (0, 0, 0)]
    cust_cmap = LinearSegmentedColormap.from_list("binary_blue", colors, N=100)

    from mpl_toolkits import mplot3d

    # df = pd.DataFrame({'Hospital Inventory Position': H_I,
    #                    'CD Inventory position': CD_I,
    #                    'Shipment Size Politic Evaluation': A
    #                    })
    # 'index', 'year', 'day', 'shipment_size', 'I0', 'I1', 'I2', 'I3', 'I4', 'reward', 'D1', 'D2',
    # 'D3',
    # 'D4',
    # 'donors', 'S1', 'S2', 'S3', 'S4', 'E1', 'E2', 'E3', 'E4', 'DC_E', 'H1_A0', 'H1_A1', 'H1_A2',
    # 'H1_A3', 'H1_A4', 'H2_A0', 'H2_A1', 'H2_A2', 'H2_A3', 'H2_A4', 'H3_A0', 'H3_A1', 'H3_A2',
    # 'H3_A3',
    # 'H3_A4', 'H4_A0', 'H4_A1', 'H4_A2', 'H4_A3', 'H4_A4', 'H1_II0', 'H1_II1', 'H1_II2', 'H1_II3',
    # 'H1_II4', 'H2_II0', 'H2_II1', 'H2_II2', 'H2_II3', 'H2_II4', 'H3_II0', 'H3_II1', 'H3_II2',
    # 'H3_II3', 'H3_II4', 'H4_II0', 'H4_II1', 'H4_II2', 'H4_II3', 'H4_II4', 'production_level'
    eval_df["Hospital Inventory Position"]=eval_df[['H1_II0', 'H1_II1', 'H1_II2', 'H1_II3',
    'H1_II4', 'H2_II0', 'H2_II1', 'H2_II2', 'H2_II3', 'H2_II4', 'H3_II0', 'H3_II1', 'H3_II2',
    'H3_II3', 'H3_II4', 'H4_II0', 'H4_II1', 'H4_II2', 'H4_II3', 'H4_II4']].sum(axis=1)
    eval_df["CD Inventory position"]=eval_df[['I0', 'I1', 'I2', 'I3', 'I4']].sum(axis=1)
    pivot = eval_df.pivot_table(index='Hospital Inventory Position', columns='CD Inventory position',
                           values='shipment_size')
    ax = sns.heatmap(pivot, cmap='viridis')
    ax.invert_yaxis()
    ax.set_title("Shipment Size Policy Evaluation")
    plt.tight_layout()
    plt.savefig('output_post/shipment_politic.png', dpi=300)
    plt.savefig('output_post/shipment_politic.svg', dpi=300)
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

    # H_I, CD_I, P = zip(*P_data)
    #
    # df = pd.DataFrame({'Hospital Inventory Position': H_I,
    #                    'CD Inventory position': CD_I,
    #                    'Production Level Politic Evaluation': P
    #                    })
    pivot = eval_df.pivot_table(index='Hospital Inventory Position', columns='CD Inventory position',
                           values='production_level')
    ax = sns.heatmap(pivot, cmap='viridis')
    ax.invert_yaxis()
    ax.set_title("Production Level Policy Evaluation")
    plt.tight_layout()
    plt.savefig('output_post/production_level.png', dpi=300)
    plt.savefig('output_post/production_level.svg', dpi=300)
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

from agent_model.training_agent_torch3 import TrainingAgent
from implementation.VMImodel import VMI
import pandas as pd
import matplotlib.pyplot as plt
import os
import concurrent.futures

if __name__ == '__main__':
    functions = ['log', 'linear', 'cos', 'gompertz','constant','consv2']

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        # print(tensorflow.test.is_gpu_available())
        train_runs = 500
        results = {executor.submit(
            TrainingAgent(model=VMI(4, 100, 5, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 5, 100), runs=train_runs,
                          steps_per_run=365,
                          batch_size=32, memory=1825, use_gpu=True,
                          epsilon_function=f, min_epsilon=0, epsilon_min_percentage=0.1).run): f for f in functions}
        rewards = {results[f]: f.result() for f in concurrent.futures.as_completed(results)}

    df = pd.DataFrame(rewards)
    df.reset_index(level=0, inplace=True)
    rollings = {}
    for func in functions:
        rollings[func] = df[func].rolling(window=50).mean()
    for func in functions:
        plt.plot(df.index, rollings[func], label=func + ' SMA(n=50)', linewidth=0.8)
    plt.legend(loc='best')
    plt.savefig('output/comparacion.png', dpi=300)
    plt.show()

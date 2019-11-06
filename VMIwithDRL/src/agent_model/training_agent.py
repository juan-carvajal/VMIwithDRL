from collections import deque
import random
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as pyplot
from matplotlib import style
import objgraph
import tensorflow



class TrainingAgent:
    def __init__(self,network_update_period, model=None, epsilon=1, epsilon_decay=0.01, min_epsilon=0, gamma=0.99, runs=100, batch_size=10,
                 steps_per_run=None):
        self.model = model
        self.memory = Memory(batch_size)
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.runs = runs
        self.step_per_run = steps_per_run
#         strategy=tensorflow.distribute.MirroredStrategy()
#         with strategy.scope():
            
        self.q_network = keras.Sequential(
            [
                keras.layers.Dense(64, input_dim=model.state_dim, activation="relu"),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(model.action_dim, activation="linear")
            ]
        )
    #         filepath="model.hdf5"
    #         checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='accuracy', verbose=0, save_best_only=True, mode='max')
    #         self.callbacks_list = [checkpoint]
        self.q_network.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

    def run(self , validateRuns=None):
        run_rewards = []
        #print(self.q_network.get_weights())
        for run in range(self.runs):
            #print("Run # ", run)
            total_reward = 0
            self.epsilon=1-(float(run)/float(self.runs))
            current_state = self.model.initial_state
            terminate = False
            train_step_count = 0
            run_step_count = 0
            while not terminate:
                action = 0
                #print("Day ",run_step_count)
                
#                 if(run_step_count % self.batch_size ==0 or True):
#                     #print(objgraph.show_most_common_types())
#                     #print(objgraph.count('tuple'))
#                     print("Day ",run_step_count)
#                     print(objgraph.show_growth(limit=10))
#                 all_objects = muppy.get_objects()
#                 sum1 = summary.summarize(all_objects)
# # Prints out a summary of the large objects
#                 summary.print_(sum1)
                if np.random.rand() <= self.epsilon:
                    # Pick random action

                    action = random.randrange(self.model.action_dim)
                    #print("Picking Random action: ",action)
                else:
                    # Best Action
                    action = np.argmax(self.q_network.predict(np.array([current_state]))[0])
                    #print("Picking Best action: ", action)
                state, action, next_state, reward, terminal = self.model.model_logic(current_state, action)
                total_reward += reward
                self.memory.append((state, action, next_state, reward, terminal))
                train_step_count += 1
                if train_step_count == self.batch_size:
                    self.replay_train()
                    train_step_count = 0
                current_state = next_state
                run_step_count += 1
                if self.step_per_run is not None and run_step_count >= self.step_per_run:
                    terminate = True
            run_rewards.append(total_reward)
            print(run, ":", total_reward,":",self.epsilon)
        style.use("ggplot")
        pyplot.scatter(range(0, len(run_rewards)), run_rewards)
        pyplot.xlabel("Run")
        pyplot.ylabel("Total Reward")
        pyplot.show()
        self.q_network.save('model.h5')
        if validateRuns:
            print("Validation Runs:")
            for i in range(validateRuns):
                total_reward = 0
                current_state = self.model.initial_state
                terminate = False
                run_step_count = 0
                while not terminate:
                    action = np.argmax(self.q_network.predict(np.array([current_state]))[0])
                    state, action, next_state, reward, terminal = self.model.model_logic(current_state, action)
                    total_reward+=reward
                    current_state=next_state
                    if terminal or run_step_count>=self.step_per_run:
                        terminate=True
                    run_step_count+=1
                print(i,":",total_reward)



    def replay_train(self):
        batch = self.memory.sample(self.batch_size)
        #x=[]
        #y=[]
        for state, action, next_state, reward, terminal in batch:
            target = reward

            if not terminal:
                target = reward + self.gamma * np.amax(self.q_network.predict(np.array([next_state]))[0])

            target_f = self.q_network.predict(np.array([state]))
            target_f[0][action] = target
            #print(target_f)
            #x.append(state)
            #y.append(target_f[0])
            #self.q_network.fit(np.array([state]), target_f, epochs=1, verbose=0)
            self.q_network.train_on_batch(np.array([state]), target_f)
            
            

        keras.backend.clear_session()
        #self.q_network.fit(np.array(x), np.array(y),batch_size=self.batch_size, epochs=1, verbose=0)

        # if self.epsilon > self.min_epsilon:
        #     self.epsilon -= self.epsilon_decay
        #     if self.epsilon < self.min_epsilon:
        #         self.epsilon = self.min_epsilon
        # else:
        #     self.epsilon = self.min_epsilon


class Memory:
    def __init__(self, max_size):
        self.memory = deque(maxlen=max_size)

    def append(self, element):
        self.memory.append(element)

    def sample(self, n):
        return random.sample(self.memory, n)
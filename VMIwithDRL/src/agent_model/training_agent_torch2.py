from collections import deque
import random
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as pyplot
from matplotlib import style
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NN(nn.Module):

    def __init__(self, model):
        super(NN, self).__init__()
        #self.norm=nn.LayerNorm(model.state_dim)
        self.l1 = nn.Linear(model.state_dim, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, 128)
        self.l4 = nn.Linear(128, model.action_dim)

    def forward(self, x):
        #x=self.norm(x)
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = F.relu(x)
        return self.l4(x)


class TrainingAgent:

    def __init__(self, model=None, min_epsilon=0.05, gamma=0.99, runs=100, batch_size=10,
                 steps_per_run=None, memory=5000, use_gpu=False, epsilon_min_percentage=0.3):
        self.model = model
        self.memory = Memory(memory)
        self.epsilon = 1
        self.batch_size = batch_size
        self.epsilon_min_percentage = epsilon_min_percentage
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.runs = runs
        self.step_per_run = steps_per_run
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.tensor = torch.cuda if self.use_gpu else torch
        self.q_network = NN(model)
        if self.use_gpu:
            print("Running model on GPU.")
            self.q_network.to("cuda")
        self.loss = nn.SmoothL1Loss()
        self.optizer = optim.Adam(self.q_network.parameters())

    def run(self , validateRuns=None):
       
        run_rewards = []
        for run in range(self.runs):
            total_reward = 0
            value = int(self.runs * self.epsilon_min_percentage)
            self.epsilon = max(self.min_epsilon, 1 - (float(run) / float(self.runs - value)))
            current_state = self.model.initial_state
            terminate = False
            run_step_count = 0
            while not terminate:
                action = 0
                if np.random.rand() <= self.epsilon:
                    action = random.randrange(self.model.action_dim)
                else:
                    qval, act = torch.max(self.q_network.forward(self.tensor.FloatTensor(current_state)), 0)
                    action = act.item()
                state, action, next_state, reward, terminal = self.model.model_logic(current_state, action)
                total_reward += reward
                self.memory.append((state, action, next_state, reward, terminal))
                if len(self.memory.memory) >= self.memory.memory.maxlen:
                    self.replay_train()
                current_state = next_state
                run_step_count += 1
                if self.step_per_run is not None and run_step_count >= self.step_per_run:
                    terminate = True
            run_rewards.append(total_reward)
            print(run, ":", total_reward, ":", self.epsilon)
#         style.use("ggplot")
#         pyplot.scatter(range(0, len(run_rewards)), run_rewards)
#         pyplot.xlabel("Run")
#         pyplot.ylabel("Total Reward")
#         pyplot.show()
        torch.save(self.q_network, "model")
        if validateRuns:
            print("Validation Runs:")
            for i in range(validateRuns):
                total_reward = 0
                current_state = self.model.initial_state
                terminate = False
                run_step_count = 0
                while not terminate:
                    qval, act = torch.max(self.q_network.forward(self.tensor.FloatTensor(current_state)), 0)
                    action = act.item()
                    state, action, next_state, reward, terminal = self.model.model_logic(current_state, action)
                    total_reward += reward
                    current_state = next_state
                    if terminal or run_step_count >= self.step_per_run:
                        terminate = True
                    run_step_count += 1
                print(i, ":", total_reward)

    def replay_train(self):
        batch = self.memory.sample(self.batch_size)
        #x=[]
        #y=[]
        for state, action, next_state, reward, terminal in batch:
            target = reward

            if not terminal:
                tensor=self.tensor.FloatTensor(next_state)
                #print(tensor)
                max,index=torch.max(self.q_network.forward(tensor),0)
                target = reward + self.gamma * max.item()
                #print(target)

            target_f = self.q_network.forward(self.tensor.FloatTensor(state))
            #print(target_f)
            #print(action)
            target_f[action] = target
            #print(target_f)
            #x.append(state)
            #y.append(target_f[0])
            #self.q_network.fit(np.array([state]), target_f, epochs=1, verbose=0)
            eval=self.q_network.forward(self.tensor.FloatTensor(state))
            self.optizer.zero_grad()
            loss=self.loss(eval,target_f)
            loss.backward()
            self.optizer.step()
#         batch = self.memory.sample(self.batch_size)
#         states = []
#         actions = []
#         next_states = []
#         rewards = []
#         terminals = []
#         
#         for state, action, next_state, reward, terminal in batch:
#             states.append(state)
#             actions.append(action)
#             next_states.append(next_state)
#             rewards.append(reward)
#             terminals.append(terminal)
#             
#         states = self.tensor.FloatTensor(states)
#         actions = self.tensor.LongTensor(actions)
#         next_states = self.tensor.FloatTensor(next_states)
#         rewards = self.tensor.FloatTensor(rewards)
#    
#         q_eval = self.q_network.forward(states).gather(1, actions.view(len(batch), 1))
#         
#         q_next = self.q_network.forward(next_states).detach()
#        
#         q_target = rewards.view(len(batch), 1) + self.gamma * q_next.max(1)[0].view(len(batch), 1)
#         self.optizer.zero_grad()
#         loss = self.loss(q_eval, q_target)
#         loss.backward()
#         self.optizer.step()


class Memory:

    def __init__(self, max_size):
        self.memory = deque(maxlen=max_size)

    def append(self, element):
        self.memory.append(element)

    def sample(self, n):
        return random.sample(self.memory, n)

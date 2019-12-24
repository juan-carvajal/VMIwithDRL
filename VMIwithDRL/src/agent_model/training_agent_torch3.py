from collections import deque
import random
from tensorflow import keras
from numpy import log as ln
import numpy as np
import matplotlib.pyplot as pyplot
from matplotlib import style
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import cos
from math import pi


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

    def __init__(self, model,runs,steps_per_run, batch_size, min_epsilon=0.05, gamma=0.99, 
                  memory=5000, use_gpu=False, epsilon_min_percentage=0.1 , epsilon_function='linear'):

        if epsilon_function == 'linear':
            self.epsilon_function=self.linear_epsilon
        elif epsilon_function=='log':
            self.epsilon_function=self.log_epsilon
        elif epsilon_function=='constant':
            self.epsilon_function=self.constant_epsilon
        elif epsilon_function=='cos':
            self.epsilon_function=self.cos_epsilon
        else:
            raise Exception('The epsilon_function parameter must be one of these types: (linear, log, constant, cos).')
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
        self.optizer = optim.Adam(self.q_network.parameters(),amsgrad =True)

    def run(self , validateRuns=None):
       
        run_rewards = []
        for run in range(self.runs):
            total_reward = 0

            epsilon=self.epsilon_function(run)
            self.model.reset_model()
            current_state = self.model.initial_state
            terminate = False
            run_step_count = 0
            while not terminate:
                action = 0
                if np.random.rand() <= epsilon:
#                     action = random.randrange(self.model.action_dim)
                    action=random.choice(self.model.valid_actions(current_state))
                else:
                    
                    q_vals=self.q_network.forward(self.tensor.FloatTensor(current_state)).cpu().detach().numpy()
                    val_actions=self.model.valid_actions(current_state)
                    max_idx=None
                    max_q=None
                    for idx, j in enumerate(q_vals):
                        if idx in val_actions:
                            if not max_q:
                                max_q=j
                                max_idx=idx
                            else:
                                if j>max_q:
                                    max_q=j
                                    max_idx=idx

#                     qval, act = torch.max(self.q_network.forward(self.tensor.FloatTensor(current_state)), 0)
#                     action = act.item()
                    action=max_idx
                state, action, next_state, reward, terminal = self.model.model_logic(current_state, action,(run,run_step_count,False))
                total_reward += reward
                self.memory.append((state, action, next_state, reward, terminal))
                if len(self.memory.memory) >= self.memory.memory.maxlen:
                    self.replay_train()
                current_state = next_state
                run_step_count += 1
                if self.step_per_run is not None and run_step_count >= self.step_per_run:
                    terminate = True
            run_rewards.append(total_reward)
            print(run, ":", total_reward, ":", epsilon)
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
                    state, action, next_state, reward, terminal = self.model.model_logic(current_state, action,(run,run_step_count,True))
                    total_reward += reward
                    current_state = next_state
                    if terminal or run_step_count >= self.step_per_run:
                        terminate = True
                    run_step_count += 1
                print(i, ":", total_reward)
        return run_rewards

    def replay_train(self):
#         batch = self.memory.sample(self.batch_size)
#         #x=[]
#         #y=[]
#         for state, action, next_state, reward, terminal in batch:
#             target = reward
# 
#             if not terminal:
#                 tensor=self.tensor.FloatTensor(next_state)
#                 #print(tensor)
#                 max,index=torch.max(self.q_network.forward(tensor),0)
#                 target = reward + self.gamma * max.item()
#                 #print(target)
# 
#             target_f = self.q_network.forward(self.tensor.FloatTensor(state))
#             #print(target_f)
#             #print(action)
#             target_f[action] = target
#             #print(target_f)
#             #x.append(state)
#             #y.append(target_f[0])
#             #self.q_network.fit(np.array([state]), target_f, epochs=1, verbose=0)
#             eval=self.q_network.forward(self.tensor.FloatTensor(state))
#             self.optizer.zero_grad()
#             loss=self.loss(eval,target_f)
#             loss.backward()
#             self.optizer.step()
        batch = self.memory.sample(self.batch_size)
        states = []
        actions = []
        next_states = []
        rewards = []
        terminals = []
         
        for state, action, next_state, reward, terminal in batch:
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            terminals.append(terminal)
             
        states = self.tensor.FloatTensor(states)
        actions = self.tensor.LongTensor(actions)
        next_states = self.tensor.FloatTensor(next_states)
        rewards = self.tensor.FloatTensor(rewards)
    
        q_eval = self.q_network.forward(states).gather(1, actions.view(len(batch), 1))
         
        q_next = self.q_network.forward(next_states).detach()
        
        q_target = rewards.view(len(batch), 1) + self.gamma * q_next.max(1)[0].view(len(batch), 1)
        self.optizer.zero_grad()
        loss = self.loss(q_eval, q_target)
        loss.backward()
        self.optizer.step()
        
        
        
    def constant_epsilon(self,run):
        return self.min_epsilon
    
    
    def linear_epsilon(self,run):
        return max(self.min_epsilon, 1 - (float(run) / float(self.runs - int(self.runs * self.epsilon_min_percentage))))
    
    def log_epsilon(self,run):
        return max(self.min_epsilon, ((self.min_epsilon -1) / ln((1-self.epsilon_min_percentage)*self.runs)) * ln(run+1) + 1      ) 

    def cos_epsilon(self,run):
        #((0.5*COS(H5*PI()/5))+0.5)*(1-(H5/350))
        c=((((1-self.min_epsilon)/2.0)*cos(run*pi/5))+(self.min_epsilon+((1-self.min_epsilon)/2.0)))*(1-(run/((1.0-self.epsilon_min_percentage)*self.runs)))
        return max(self.min_epsilon,c )
    
    
    
    
class Memory:

    def __init__(self, max_size):
        self.memory = deque(maxlen=max_size)

    def append(self, element):
        self.memory.append(element)

    def sample(self, n):
        return random.sample(self.memory, n)

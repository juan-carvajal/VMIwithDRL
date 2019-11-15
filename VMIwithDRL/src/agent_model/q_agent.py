import numpy as np
import random

class QAgent:
    def __init__(self , model , gamma , learning_rate ,min_epsilon=0.01, epsilon_min_percentage=0.3):
        self.model=model
        self.q_table={}
        self.epsilon=1
        self.gamma=gamma
        self.learning_rate=learning_rate
        self.min_epsilon=min_epsilon
        self.epsilon_min_percentage=epsilon_min_percentage
        
        
    def run(self,steps_per_run, runs):
        run_rewards = []
        for run in range(runs):
            total_reward = 0
            value = int(runs * self.epsilon_min_percentage)
            self.epsilon = max(self.min_epsilon, 1 - (float(run) / float(runs - value)))
            current_state = self.model.initial_state
            terminate = False
            run_step_count = 0
            while not terminate:
                action = 0
                t_current=tuple(current_state)
                if not t_current in self.q_table:
                    self.q_table[t_current]=[0]*self.model.action_dim
                if np.random.rand() <= self.epsilon:
                    action = random.randrange(self.model.action_dim)
                else:
                    action = np.argmax(self.q_table[t_current])

                        
                        
                state, action, next_state, reward, terminal = self.model.model_logic(current_state, action)
                total_reward += reward
                
                t_next=tuple(next_state)
                if t_next in self.q_table:
                    value=np.argmax(self.q_table[t_next])
                else:
                    value=0
                self.q_table[t_current][action]=self.q_table[t_current][action]+self.learning_rate*(reward+(self.gamma*value) - self.q_table[t_current][action])
                
                current_state = next_state
                run_step_count += 1
                if steps_per_run is not None and run_step_count >= steps_per_run:
                    terminate = True
            run_rewards.append(total_reward)
            print(run, ":", total_reward, ":", self.epsilon)
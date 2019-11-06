from agent_model.model import Model
from agent_model.training_agent import TrainingAgent
from implementation.hospital import Hospital
import numpy as np
from optimizer.AllocationOptimizerCplexDocPlex import AllocationOptimizer





class VMI(Model):
    # LOS ESTADOS DE ESTA CLASE SON LOS NIVELES DE INVENTARIOS QUE TIENE PARA CADA UNA DE LAS CADUCIDADES
    # LAS ACCIONES POSIBLES VAN DESDE 0 HASTA MAX_A
    def __init__(self, hospitals, max_A, shelf_life, initial_state=None, exp_cost=None, stockout_cost=None):
        super(VMI, self).__init__(initial_state, max_A, shelf_life)
        self.day = 1
        self.shelf_life = shelf_life
        self.exp_cost = exp_cost
        self.stockout_cost = stockout_cost
        self.hospitals = [Hospital([0] * shelf_life, None, exp_cost, stockout_cost)] * hospitals


    def model_logic(self, state, action):
        demands = [5, 10, 15, 20]
        donors = 100;
        A = action
        A_i = [0] * self.shelf_life
        for i, val in enumerate(A_i):
            if (i == 0):
                A_i[i] = min(A, state[i])
            else:
                A_i[i] = min(A - sum(A_i[0:i]), state[i])

        II = []
        for i in self.hospitals:
            II.append(i.inventory)

        # II = [[0, 2, 0, 3, 0], [0, 1, 0, 3, 3], [0, 1, 2, 2, 5], [0, 3, 5, 1, 1]]
        # D = [5, 5, 5, 5]
        # A = [6, 7, 8, 9, 10]
        # M = 1000000
        # CF = 100
        # CV = 10
        # R = 5
        # H = 4
        # a = AllocationOptimizer(II, A, D, CV, CF, R, H)
        # x = a.allocate()
        # print(x)
        # print(x[0])

        # print(II)
        # print(A_i)
        # print(demands)
        # print(self.exp_cost)
        # print(self.stockout_cost)
        # print(self.shelf_life)
        # print(len(self.hospitals))

        opt = AllocationOptimizer(II, A_i, demands, self.exp_cost, self.stockout_cost, self.shelf_life,
                                  len(self.hospitals))

        # opt = AllocationOptimizer(II, A_i, demands, self.exp_cost, self.stockout_cost, self.shelf_life, len(self.hospitals))
        rep = opt.allocate()

        #print(rep)
        next_state = self.update_inventory_bloodbank(state, donors, action)
        # print(donors)
        # print(next_state)

        reward = state[0] * self.exp_cost
        #print(reward)
        for hosp in range(len(self.hospitals)):
            reward += self.hospitals[hosp].supply(rep[hosp], demands[hosp])
        # print(reward)
        return state, action, next_state, -reward, False

    def update_inventory_bloodbank(self, state, donors, action):
        state_aux = [0] * len(state)
        for i in range(self.shelf_life):
            if (i == 0):
                state_aux[i] = max(0, state[i + 1] - action)
            elif 0 < i < 4:
                state_aux[i] = max(0, state[i + 1] - max(0, action - sum(state[:i])))
            else:
                state_aux[i] = max(0, donors - max(0, action - sum(state[:i])))

        state = state_aux;

        return state


initial_state = [0, 0, 0, 0, 0]
#print(tensorflow.test.is_gpu_available())
model = VMI(4, 100, 5, initial_state, 5, 100)
agent = TrainingAgent(model=model, runs=100, steps_per_run=365, batch_size=32, epsilon_decay=0.01,network_update_period=10)
agent.run(validateRuns=10
          )

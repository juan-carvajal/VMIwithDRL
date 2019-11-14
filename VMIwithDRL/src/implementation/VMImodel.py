from agent_model.model import Model
from agent_model.training_agent_torch2 import TrainingAgent
from implementation.hospital import Hospital
import numpy as np
from optimizer.AllocationOptimizerCplexDocPlex import AllocationOptimizer
import timeit
import math
from scipy import stats




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
        #demands = [5, 10, 15, 20]
        donors = 100
        demands = self.get_demand()
        #donors = self.get_donors()
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
        reward*=-1
        return state, action, next_state, reward, False

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

    def get_donors(self):
        
        #FOR MONDAYS
        mu = 109
        desv = 44.36799
        don = np.random.normal(mu, desv, 1)
        don = math.floor(don)
        #don=100
        return don
    
    
    
    def get_demand(self):
        #DISTRIBUTION FOR MONDAYS
        #VENTA DIRECTA UNION TEMPORAL
        m = 1.33950
        c = 1.29446
        d1 = np.random.lognormal(m,c,1)
        d1 = self.checkDemand(d1)
        #HOSPIAL DE SUBA
        mu2 =9.35
        desv2 = 6.29428
        d2 = np.random.normal(mu2, desv2, 1)
        d2 = self.checkDemand(d2)
        #HOSPITAL SANTA CLARA
        mu3 = 11.93478
        desv3 = 4.98621
        d3 = np.random.normal(mu3, desv3, 1)
        d3 = self.checkDemand(d3)
        #MIOCARDIO SAS
        m1 = 1.85161
        c1= 5.46852
        d4 = np.random.lognormal(m1,c1,1)
        d4 = self.checkDemand(d4)
        #demands = [d1,d2,d3,d4]
        demands=[10,15,8,11]
        return demands
    
    def checkDemand(self, a):
        a = math.floor(a)
        if(a == 0):
            a = 1
        return a

        


initial_state = [0, 0, 0, 0, 0]
#print(tensorflow.test.is_gpu_available())
model = VMI(4, 100, 5, initial_state, 5, 100)
agent = TrainingAgent(model=model, runs=500, steps_per_run=365, batch_size=500,memory=10000,use_gpu=True)

agent.run(validateRuns=10
           )
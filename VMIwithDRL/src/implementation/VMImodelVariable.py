from implementation.hospital import Hospital
import numpy as np
from implementation.optimizer.AllocationOptimizerHeuristica import AllocationOptimizer
#from implementation.optimizer.AllocationOptimizerGoalProgramming3 import AllocationOptimizer
# from implementation.optimizer.AllocationOptimizerNonGoal import AllocationOptimizer
from collections import deque
from agent_model.model import Model
# from optimizer.AllocationOptimizerCplexDocPlex import AllocationOptimizer
from sklearn.metrics import mean_squared_error
import timeit
import math
import pandas as pd
from implementation.inventory import FIFOInventory
from statistics import mean
import json
from scipy import stats


class VMI(Model):
    # LOS ESTADOS DE ESTA CLASE SON LOS NIVELES DE INVENTARIOS QUE TIENE PARA CADA UNA DE LAS CADUCIDADES
    # LAS ACCIONES POSIBLES VAN DESDE 0 HASTA MAX_A
    def __init__(self, hospitals, max_A, shelf_life, train_runs, initial_state=None, exp_cost=None, stockout_cost=None):
        super(VMI, self).__init__(initial_state, max_A * 11, len(initial_state))
        self.year_day = 0
        self.year = 0
        self.day = 1
        self.train_runs = train_runs
        self.shelf_life = shelf_life
        self.exp_cost = exp_cost
        self.stockout_cost = stockout_cost
        self.hospitals = [Hospital([0] * shelf_life, 1.5 * exp_cost, stockout_cost) for _ in range(hospitals)]
        # [Hospital([0] * shelf_life, None, exp_cost*1.5, stockout_cost*1.5)] * hospitals
        self.demands_and_donors = pd.read_csv(r'implementation/run_parameters.csv')
        # print(self.demands_and_donors)
        self.demand_registry = [deque(maxlen=3) for _ in range(hospitals)]
        self.log = {"train": {}, "validate": {}}
        self.state_space_memory = []
        for state in initial_state:
            self.state_space_memory.append({state})

    def model_logic(self, state, action):
        # demands = [5, 10, 15, 20]
        # print(state[:self.shelf_life])
        # demand_data =self.get_demand(state[5])# self.demands_and_donors.iloc[self.year_day]

        # donors = demand_data["donors"]
        for index, value in enumerate(state):
            if value not in self.state_space_memory[index]:
                self.state_space_memory[index].add(value)
        donors = 100
        # self.get_donors(state[5])
        demands = self.get_demand(state[5])
        # self.get_demand(state[5])
        # donors = self.get_donors()
        # A = min(action,sum(state[:self.shelf_life]))
        A = action // 11
        prep_donors = int((((action % 11) * 10) / 100.0) * donors)

        # print(action, A, prep_donors ,sum(demands))

        A_i = [0] * self.shelf_life
        for i, val in enumerate(A_i):
            if i == 0:
                A_i[i] = min(A, state[i])
            else:
                A_i[i] = min(A - sum(A_i[0:i]), state[i])

        II = []
        for i in self.hospitals:
            II.append(i.inventory.inventory)

        # demand_forecast = self.get_average_demand(state[5])
        demand_forecast = demands

        opt = AllocationOptimizer(II, A_i, demand_forecast, self.exp_cost, self.stockout_cost, self.shelf_life,
                                  len(self.hospitals))
        rep, used_model = opt.allocate()
        for idx, i in enumerate(self.demand_registry):
            i.append(demands[idx])
        # opt = AllocationOptimizer(II, A_i, demands, self.exp_cost, self.stockout_cost, self.shelf_life, len(self.hospitals))

        # print("Day ",self.year_day, rep)

        # print(rep)
        reward = 0
        rewards = []
        stockouts = []
        expireds = []

        for hosp in range(len(self.hospitals)):
            r, st, exp = self.hospitals[hosp].supply(rep[hosp], demands[hosp])
            rewards.append(-r)
            stockouts.append(st)
            expireds.append(exp)
            reward += r

        next_state, dc_exp = self.update_inventory_bloodbank(state, prep_donors, A,
                                                             [i.inventory.inventory for i in self.hospitals])
        # print(donors)
        # print(next_state)

        reward += dc_exp * self.exp_cost
        # reward=0
        # print(reward)
        year = self.year
        if year < self.train_runs:
            data = {"rewards": rewards, "stockouts": stockouts, "expirees": expireds, "allocation": rep,
                    "shipment_size": A, "production_level": (((action % 11) * 10) / 100.0),
                    "inventory": state[:self.shelf_life], "donors": donors, "reward": reward, "demands": demands,
                    'DC_expirees': dc_exp, 'II': II, 'Used_LP_Model': used_model}
            if year in self.log["train"]:
                self.log["train"][year].append(data)
            else:
                self.log["train"][year] = []
                self.log["train"][year].append(data)
        else:
            data = {"rewards": rewards, "stockouts": stockouts, "expirees": expireds, "allocation": rep,
                    "shipment_size": A, "production_level": (((action % 11) * 10) / 100.0),
                    "inventory": state[:self.shelf_life], "donors": donors, "reward": reward, "demands": demands,
                    'DC_expirees': dc_exp, 'II': II, 'Used_LP_Model': used_model}
            if year in self.log["validate"]:
                self.log["validate"][year].append(data)
            else:
                self.log["validate"][year] = []
                self.log["validate"][year].append(data)

        self.year_day += 1
        # print(reward)
        reward *= -1
        return state, action, next_state, reward, False

    def valid_actions(self, state):
        t_inv = sum(state[:self.shelf_life])
        # a_max = min(t_inv, self.action_dim)
        # v_act = [*range(a_max)]

        v_act2 = {x for x in range(1100) if (x // 11) <= t_inv}
        # print(t_inv,v_act2)
        return v_act2

    def reset_model(self):
        # print("Solutions buffer:",len(self.solve_memory))
        self.forecast_acc_mse = 0
        self.year_day = 0
        self.year += 1
        self.hospitals = [Hospital([0] * self.shelf_life, self.exp_cost * 1.5, self.stockout_cost) for _ in
                          range(len(self.hospitals))]
        for i in self.demand_registry:
            i.clear()

    def update_inventory_bloodbank(self, state, donors, delivered, hospital_new_inv):
        inv = FIFOInventory(state[:self.shelf_life])
        stk = inv.pop(delivered)
        dc_exp = inv.age()
        supply = [0] * self.shelf_life
        supply[-1] = donors
        inv.push(supply)

        if stk > 0:
            raise Exception("Malfunction : DC should never incur in stockouts. ")
        # state_aux = [0] * (self.shelf_life)
        # dc_exp = state[0]
        # for i in range(self.shelf_life):
        #     if (i == 0):
        #         state_aux[i] = max(0, state[i + 1] - delivered)
        #     elif 0 < i < 4:
        #         state_aux[i] = max(0, state[i + 1] - max(0, delivered - sum(state[:i])))
        #     elif (i == 4):
        #         state_aux[i] = max(0, donors - max(0, delivered - sum(state[:i])))
        #
        # state_aux[5] = (state[5] % 7) + 1
        #
        # state_aux += hospital_new_inv
        state_aux = inv.inventory
        state_aux += [(state[5] % 7) + 1]
        state_aux += [item for sublist in hospital_new_inv for item in sublist]
        return state_aux, dc_exp

    def arima_forecast(self):
        import pmdarima as pm
        forecast = [round(pm.auto_arima(self.demand_registry[i],
                                        start_p=1,
                                        start_q=1,
                                        test="adf",
                                        seasonal=True,
                                        trace=False).predict(n_periods=1, return_conf_int=False)[0]) for i in
                    range(len(self.hospitals))]
        return forecast

    def get_donors(self, day):

        if day == 1:
            don = np.random.triangular(50, 90, 120)

        elif day == 2:
            don = np.random.triangular(50, 90, 120)

        elif day == 3:
            don = np.random.triangular(50, 90, 120)

        elif day == 4:
            don = np.random.triangular(50, 90, 120)

        elif day == 5:
            don = np.random.triangular(50, 90, 120)

        elif day == 6:
            don = np.random.triangular(50, 90, 120)
        else:
            don = np.random.triangular(50, 90, 120)

        don = math.floor(don)
        # don=100
        return don

    def get_average_demand(self, day):
        if day == 1:
            d1 = 2.6 * 6.1 * 0.5
            d2 = 2.6 * 6.1 * 0.3
            d3 = 2.6 * 6.1 * 0.2
            d4 = 2.6 * 6.1 * 0.1

        elif day == 2:
            d1 = 4.9 * 9.2 * 0.5
            d2 = 4.9 * 9.2 * 0.3
            d3 = 4.9 * 9.2 * 0.2
            d4 = 4.9 * 9.2 * 0.1

        elif day == 3:
            d1 = 6.9 * 8.2 * 0.5
            d2 = 6.9 * 8.2 * 0.3
            d3 = 6.9 * 8.2 * 0.2
            d4 = 6.9 * 8.2 * 0.1

        elif day == 4:
            d1 = 4.7 * 9.3 * 0.5
            d2 = 4.7 * 9.3 * 0.3
            d3 = 4.7 * 9.3 * 0.2
            d4 = 4.7 * 9.3 * 0.1

        elif day == 5:
            d1 = 5.7 * 8.0 * 0.5
            d2 = 5.7 * 8.0 * 0.3
            d3 = 5.7 * 8.0 * 0.2
            d4 = 5.7 * 8.0 * 0.1

        elif day == 6:
            d1 = 4.8 * 8.7 * 0.5
            d2 = 4.8 * 8.7 * 0.3
            d3 = 4.8 * 8.7 * 0.2
            d4 = 4.8 * 8.7 * 0.1
        else:
            d1 = 1.7 * 3.2 * 0.5
            d2 = 1.7 * 3.2 * 0.3
            d3 = 1.7 * 3.2 * 0.2
            d4 = 1.7 * 3.2 * 0.1
        d1 = self.checkDemand(d1)
        d2 = self.checkDemand(d2)
        d3 = self.checkDemand(d3)
        d4 = self.checkDemand(d4)
        return [d1, d2, d3, d4]

    def get_demand(self, day):
        # VENTA DIRECTA UNION TEMPORAL

        if day == 1:
            d1 = np.random.gamma(2.6, 6.1) * 0.5
            d2 = np.random.gamma(2.6, 6.1) * 0.3
            d3 = np.random.gamma(2.6, 6.1) * 0.2
            d4 = np.random.gamma(2.6, 6.1) * 0.1

        elif day == 2:
            d1 = np.random.gamma(4.9, 9.2) * 0.5
            d2 = np.random.gamma(4.9, 9.2) * 0.3
            d3 = np.random.gamma(4.9, 9.2) * 0.2
            d4 = np.random.gamma(4.9, 9.2) * 0.1

        elif day == 3:
            d1 = np.random.gamma(6.9, 8.2) * 0.5
            d2 = np.random.gamma(6.9, 8.2) * 0.3
            d3 = np.random.gamma(6.9, 8.2) * 0.2
            d4 = np.random.gamma(6.9, 8.2) * 0.1

        elif day == 4:
            d1 = np.random.gamma(4.7, 9.3) * 0.5
            d2 = np.random.gamma(4.7, 9.3) * 0.3
            d3 = np.random.gamma(4.7, 9.3) * 0.2
            d4 = np.random.gamma(4.7, 9.3) * 0.1

        elif day == 5:
            d1 = np.random.gamma(5.7, 8.0) * 0.5
            d2 = np.random.gamma(5.7, 8.0) * 0.3
            d3 = np.random.gamma(5.7, 8.0) * 0.2
            d4 = np.random.gamma(5.7, 8.0) * 0.1

        elif day == 6:
            d1 = np.random.gamma(4.8, 8.7) * 0.5
            d2 = np.random.gamma(4.8, 8.7) * 0.3
            d3 = np.random.gamma(4.8, 8.7) * 0.2
            d4 = np.random.gamma(4.8, 8.7) * 0.1
        else:
            d1 = np.random.gamma(1.7, 3.2) * 0.5
            d2 = np.random.gamma(1.7, 3.2) * 0.3
            d3 = np.random.gamma(1.7, 3.2) * 0.2
            d4 = np.random.gamma(1.7, 3.2) * 0.1

        d1 = self.checkDemand(d1)
        d2 = self.checkDemand(d2)
        d3 = self.checkDemand(d3)
        d4 = self.checkDemand(d4)

        demands = [d1, d2, d3, d4]
        # demands=[10,15,8,11]
        return demands

    def checkDemand(self, a):
        a = math.floor(a)
        if (a == 0):
            a = 1
        return a

# agent=QAgent(model,0.99,0.1)
# agent.run(365, 1000)

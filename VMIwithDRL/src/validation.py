import numpy as np
from implementation.inventory import FIFOInventory
from statistics import mean
import math
import random
import pandas as pd


class ValidationModel:

    def __init__(self, parameters, years_to_run=50, exp_cost=5, stk_cost=100):
        self.inventory = FIFOInventory([10] * 5)
        self.hosp_inv = [FIFOInventory([0] * 5) for i in range(4)]
        self.week_day = 1
        self.parameters = parameters
        self.years_to_run = years_to_run
        self.exp_cost = exp_cost
        self.stk_cost = stk_cost
        self.log = {"year": []}

    def reset(self):
        self.inventory = FIFOInventory([10] * 5)
        self.hosp_inv = [FIFOInventory([0] * 5) for i in range(4)]
        self.week_day = 1

    def run(self):
        dc_exps = []
        stks = []
        exps = []
        rewards = []
        super_log = []
        for year in range(self.years_to_run):
            last_day_orders = [0] * 4
            self.reset()
            dc_exp = 0
            stk = 0
            exp = 0
            for day in range(365):
                log_data = [year, day]
                log_data += self.inventory.inventory
                for i in self.hosp_inv:
                    log_data += i.inventory
                log_data += last_day_orders
                stk_ = 0
                to_send = self.shipping_heuristic(self.inventory.inventory, last_day_orders)
                # stk_+=max(sum(last_day_orders)-sum(to_send),0)
                demands = self.get_demand(self.week_day)

                log_data += to_send
                log_data += demands

                exp_ = 0
                for hosp_idx in range(len(self.hosp_inv)):
                    ship_distribution = self.fifo_ship(self.inventory.inventory, to_send[hosp_idx])
                    self.inventory.pop(to_send[hosp_idx])
                    self.hosp_inv[hosp_idx].push(ship_distribution)
                    stk_ += self.hosp_inv[hosp_idx].pop(demands[hosp_idx])
                    exp_ += self.hosp_inv[hosp_idx].age()
                    last_day_orders[hosp_idx] = self.inventory_rule(self.hosp_inv[hosp_idx].inventory,
                                                                    self.parameters[hosp_idx + 1])
                stk += stk_
                exp += exp_
                dc_exp_ = self.inventory.age()
                dc_exp += dc_exp_
                new_donors = [0] * 5
                new_donors[4] = self.inventory_rule(self.inventory.inventory, self.parameters[0])
                log_data += [new_donors[4]]
                log_data += [self.week_day]
                self.inventory.push(new_donors)
                self.week_day = (self.week_day % 7) + 1
                log_data += [dc_exp_, exp_, stk_]
                super_log.append(log_data)
            reward = dc_exp * self.exp_cost + exp * self.exp_cost * 1.5 + stk * self.stk_cost
            rewards.append(reward)
            dc_exps.append(dc_exp)
            exps.append(exp)
            stks.append(stk)
            columns = ['Year', 'Day', 'II_1', 'II_2', 'II_3', 'II_4', 'II_5', 'HI_1_1', 'HI_1_2', 'HI_1_3', 'HI_1_4',
                       'HI_1_5', 'HI_2_1', 'HI_2_2', 'HI_2_3', 'HI_2_4', 'HI_2_5', 'HI_3_1', 'HI_3_2', 'HI_3_3',
                       'HI_3_4', 'HI_3_5', 'HI_4_1', 'HI_4_2', 'HI_4_3', 'HI_4_4', 'HI_4_5', 'ORDERS_1', 'ORDERS_2',
                       'ORDERS_3', 'ORDERS_4', 'REAL_ORDER_1', 'REAL_ORDER_2', 'REAL_ORDER_3', 'REAL_ORDER_4',
                       'DEMAND_1', 'DEMAND_2', 'DEMAND_3', 'DEMAND_4', 'CD_ORDER', 'WEEKDAY', 'DC_EXP', 'EXP', 'STK'
                       ]
        return -mean(rewards), mean(dc_exps), mean(exps), mean(stks), pd.DataFrame(super_log, columns=columns)

    def fifo_ship(self, inventory, shipment_size):
        if (sum(inventory) < shipment_size):
            raise Exception("Shipment size can't exceed inventory")

        shipment = [0] * len(inventory)
        for i, val in enumerate(shipment):
            if i == 0:
                shipment[i] = min(shipment_size, inventory[i])
            else:
                shipment[i] = min(shipment_size - sum(shipment[0:i]), inventory[i])
        return shipment

    def inventory_rule(self, inventory, S):
        total_inv = sum(inventory)
        if total_inv >= S * 10:
            return 0
        else:
            #return self.round_to_multiple(S * 10 - total_inv,10,ceil=True)
            #return 10 * math.ceil((S * 10 - total_inv) / 10)
            return S-total_inv

    def shipping_heuristic(self, inventory, orders):
        total_inv = sum(inventory)
        if total_inv == 0:
            return [0] * len(orders)
        total_orders = sum(orders)
        val = (total_inv * 0.56) / 50
        if (val >= 1):
            heuristic = 1
        else:
            heuristic = val
        aux = heuristic * total_orders

        if (aux <= total_inv):
            service_level = heuristic
        else:
            service_level = total_inv / total_orders
        #ship_orders = [self.round_to_multiple(int(service_level * order),10) for order in orders]
        ship_orders = [int(service_level * order) for order in orders]
        #print(service_level,orders,ship_orders)
        return ship_orders

    def round_to_multiple(self, n, multiple, ceil=False):
        if ceil:
            return multiple * math.ceil(n / multiple)
        else:
            return multiple * math.floor(n / multiple)

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


def eval(individual):
    test = ValidationModel(individual)
    results = test.run()
    return results[0],


def createIndividual():
    return [random.randint(0, 100) for _ in range(5)]


if __name__ == '__main__':
    test = ValidationModel([52, 84, 51, 31, 16], years_to_run=1)
    a = test.run()
    print(a[:-1])
    df = a[-1]
    print(df)
    df.to_csv('validation_data_heuristic.csv')
    # from deap import base
    # from deap import creator
    # from deap import tools
    # from deap import algorithms
    #
    # creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # creator.create("Individual", list, fitness=creator.FitnessMax)
    # toolbox = base.Toolbox()
    # toolbox.register("indices", createIndividual)
    # toolbox.register("individual", tools.initIterate, creator.Individual,
    #                  toolbox.indices)
    # toolbox.register("mate", tools.cxTwoPoint)
    # toolbox.register("mutate", tools.mutUniformInt,low=0,up=100,indpb=0.25)
    # toolbox.register("select", tools.selTournament, tournsize=5)
    # toolbox.register("evaluate", eval)
    # toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # pop = toolbox.population(n=10)
    #
    # hof = tools.HallOfFame(1)
    # stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", np.mean)
    # stats.register("std", np.std)
    # stats.register("min", np.min)
    # stats.register("max", np.max)
    #
    # algorithms.eaSimple(pop, toolbox, 0.25, 0.25, 100, stats=stats,
    #                     halloffame=hof, verbose=True)
    # print(hof)

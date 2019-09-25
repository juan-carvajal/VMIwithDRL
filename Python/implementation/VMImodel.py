from agent_model.model import Model
from agent_model.training_agent import TrainingAgent
from implementation.hospital import Hospital
import numpy as np
from optimizer.AllocationOptimizer import AllocationOptimizer


class VMI(Model):
    # LOS ESTADOS DE ESTA CLASE SON LOS NIVELES DE INVENTARIOS QUE TIENE PARA CADA UNA DE LAS CADUCIDADES
    # LAS ACCIONES POSIBLES VAN DESDE 0 HASTA MAX_A
    def __init__(self, hospitals, max_A, shelf_life, initial_state=None, exp_cost=None, stockout_cost=None):
        super(VMI, self).__init__(initial_state, max_A, shelf_life)
        self.day = 1
        self.shelf_life = shelf_life
        self.exp_cost = exp_cost
        self.stockout_cost = stockout_cost
        self.hospitals = [Hospital([0] * 5, None, exp_cost, stockout_cost)] * hospitals

    def model_logic(self, state, action):
        demands = [5, 10, 15, 20]
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

        opt = AllocationOptimizer(II, A_i, demands, self.exp_cost, self.stockout_cost, self.shelf_life,
                                  len(self.hospitals))
        rep = opt.allocate()

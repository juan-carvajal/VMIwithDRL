# Import PuLP modeler functions
from pulp import *
import numpy as np


class AllocationOptimizer():
    def __init__(self, II, A, D, CV, CF,R,H):
        self.II = II
        self.A = A
        self.D = D
        self.CV = CV
        self.CF = CF
        self.M = 1000000
        self.R=list(range(R))
        self.H=list(range(H))

    def allocate(self):
        # self.R = list(range(5))
        # H = list(range(4))
        RX = list(range(len(self.R)))

        prob = LpProblem("LPOptimizationProblem", LpMinimize)

        x = pulp.LpVariable.matrix("x", (self.H, RX), 0, None, LpInteger)
        I0 = pulp.LpVariable.dicts("I0", self.H, 0, None, LpInteger)
        F = pulp.LpVariable.dicts("Fh", self.H, 0, None, LpInteger)
        YI0 = pulp.LpVariable.dicts("YI0h", self.H, 0, 1, LpInteger)
        YF = pulp.LpVariable.dicts("YFh", self.H, 0, 1, LpInteger)
        
        cons5_plus = pulp.LpVariable.dicts("Constraint5plus", self.H, 0, None, LpContinuous)
        cons5_minus = pulp.LpVariable.dicts("Constraint5minus", self.H, 0, None, LpContinuous)
        cons7_plus = pulp.LpVariable.dicts("Constraint7plus", self.H, 0, None, LpContinuous)
        cons7_minus = pulp.LpVariable.dicts("Constraint7minus", self.H, 0, None, LpContinuous)

        coeff=[1,1,1,1]


        
        cov = coeff[0]*lpSum([I0[h] * self.CV for (h) in self.H])
        cof = coeff[0]*lpSum([F[h] * self.CF for (h) in self.H])
        cor5_plus = coeff[1]*lpSum(cons5_plus[h] for (h) in self.H)
        cor7_plus = coeff[2]*lpSum(cons7_plus[h] for (h) in self.H)
        cor7_minus = coeff[3]*lpSum(cons7_minus[h] for (h) in self.H)
        # The objective function is added to 'prob' first
        
        prob += cov + cof + cor5_plus + cor7_plus + cor7_minus

        for h in self.H:
            prob += -self.II[h][1] - x[h][0] + self.D[h] <= self.M * YI0[h]

        for h in self.H:
            prob += self.II[h][1] + x[h][0] - self.D[h] <= self.M * (1 - YI0[h])

        for h in self.H:
            prob += I0[h] >= 0

        for h in self.H:
            prob += I0[h] >= self.II[h][1] + x[h][0] - self.D[h]

        for h in self.H:
            prob += I0[h] <= self.M * (1 - YI0[h])

        for h in self.H:
            prob += I0[h] <= self.II[h][1] + x[h][0] - self.D[h] + self.M * YI0[h]

            ########################################

        for h in self.H:
            prob += -self.D[h] + lpSum([self.II[h][r] for (r) in self.R]) + lpSum([x[h][r] for (r) in self.R]) <= self.M * YF[h]

        for h in self.H:
            prob += self.D[h] - lpSum([self.II[h][r] for (r) in self.R]) - lpSum([x[h][r] for (r) in self.R]) <= self.M * (1 - YF[h])

        for h in self.H:
            prob += F[h] >= 0

        for h in self.H:
            prob += F[h] >= self.D[h] - lpSum([self.II[h][r] for (r) in self.R]) - lpSum([x[h][r] for (r) in self.R]), "Const" + str(h)

        for h in self.H:
            prob += F[h] <= self.M * (1 - YF[h])

        for h in self.H:
            prob += F[h] <= self.D[h] - lpSum([self.II[h][r] for (r) in self.R]) - lpSum([x[h][r] for (r) in self.R]) + self.M * YF[h]

        for h in self.H:
            prob += -cons5_plus[h]+cons5_minus[h] + F[h] - (1/len(self.H)) * lpSum([F[h1] for (h1) in self.H]) == 0 

        for r in self.R:
            prob += self.A[r] == lpSum([x[h][r] for (h) in self.H])

            
        for h in self.H[:-1]:    
            prob += (((lpSum([x[h][r] for (r) in self.R]) / self.D[h]) -  (lpSum([x[h+1][r] for (r) in self.R]) / self.D[h+1]) -
                     cons7_plus[h]+cons7_minus[h] ==0))
            

        # The problem data is written to an .lp file
       

        # The problem is solved using PuLP's choice of Solver
        prob.solve()
        

        # The status of the solution is printed to the screen
        #print("Status:", LpStatus[prob.status])
        a = [[0 for r in range(len(self.R))] for h in range(len(self.H))]
        
        for h in self.H:
            for r in self.R:
                #print ("x" + str(h) + str(r), x[h][r].varValue)
                a[h][r] =  x[h][r].varValue
 
        #The optimised objective function value is printed to the screen
        #print(value(prob.objective))  
        
        
        
        
        return a, True



if __name__ == '__main__':
    
    # II = [[0, 2, 0, 3, 0], [0, 1, 0, 3, 3], [0, 1, 2, 2, 5], [0, 3, 5, 1, 1]]
    # D = [5, 5, 5, 5]
    # A = [6, 7, 8, 9, 10]

    II = [[9, 0, 4, 0, 0], [0, 0, 21, 0, 21], [0, 0, 13, 5, 9], [0, 0, 23, 0, 0]]
    D = [15, 11, 2, 9]
    A = [0, 0, 0, 35, 21]

    M = 1000000
    CF = 100
    CV = 7.5
    R = 5
    H = 4

    a = AllocationOptimizer(II, A, D, CV, CF, R, H)
    x = a.allocate()

    print(x)

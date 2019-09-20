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
        RX = list(range(self.R.__len__()))

        prob = LpProblem("LPOptimizationProblem", LpMinimize)

        x = pulp.LpVariable.matrix("x", (RX, self.H), 0, None, LpInteger)
        I0 = pulp.LpVariable.dicts("I0", self.H, 0, None, LpInteger)
        F = pulp.LpVariable.dicts("Fh", self.H, 0, None, LpInteger)
        YI0 = pulp.LpVariable.dicts("YI0h", self.H, 0, 1, LpInteger)
        YF = pulp.LpVariable.dicts("YFh", self.H, 0, 1, LpInteger)

        cov = lpSum([I0[h] * self.CV for (h) in self.H])
        cof = lpSum([F[h] * self.CF for (h) in self.H])

        # The objective function is added to 'prob' first
        prob += cov + cof

        for h in self.H:
            prob += -self.II[1][h] - x[0][h] + self.D[h] <= self.M * YI0[h]

        for h in self.H:
            prob += self.II[1][h] + x[0][h] - self.D[h] <= self.M * (1 - YI0[h])

        for h in self.H:
            prob += I0[h] >= 0

        for h in self.H:
            prob += I0[h] >= self.II[1][h] + x[0][h] - self.D[h]

        for h in self.H:
            prob += I0[h] <= self.M * (1 - YI0[h])

        for h in self.H:
            prob += I0[h] <= self.II[1][h] + x[0][h] - self.D[h] + self.M * YI0[h]

            ########################################

        for h in self.H:
            prob += -self.D[h] + lpSum([self.II[r][h] for (r) in self.R]) + lpSum([x[r][h] for (r) in self.R]) <= self.M * YF[h]

        for h in self.H:
            prob += self.D[h] - lpSum([self.II[r][h] for (r) in self.R]) - lpSum([x[r][h] for (r) in self.R]) <= self.M * (1 - YF[h])

        for h in self.H:
            prob += F[h] >= 0

        for h in self.H:
            prob += F[h] >= self.D[h] - lpSum([self.II[r][h] for (r) in self.R]) - lpSum([x[r][h] for (r) in self.R]), "Const" + str(h)

        for h in self.H:
            prob += F[h] <= self.M * (1 - YF[h])

        for h in self.H:
            prob += F[h] <= self.D[h] - lpSum([self.II[r][h] for (r) in self.R]) - lpSum([x[r][h] for (r) in self.R]) + self.M * YF[h]

        for h in self.H:
            prob += F[h] <= 0.25 * lpSum([F[h1] for (h1) in self.H])

        for r in self.R:
            prob += self.A[r] == lpSum([x[r][h] for (h) in self.H])

        prob += (lpSum([self.II[r][0] for (r) in self.R]) + lpSum([x[r][0] for (r) in self.R])) / self.D[0] == (
                lpSum([self.II[r][1] for (r) in self.R]) + lpSum([x[r][1] for (r) in self.R])) / self.D[1]
        prob += (lpSum([self.II[r][1] for (r) in self.R]) + lpSum([x[r][1] for (r) in self.R])) / self.D[1] == (
                lpSum([self.II[r][2] for (r) in self.R]) + lpSum([x[r][2] for (r) in self.R])) / self.D[2]
        prob += (lpSum([self.II[r][2] for (r) in self.R]) + lpSum([x[r][2] for (r) in self.R])) / self.D[2] <= (
                lpSum([self.II[r][3] for (r) in self.R]) + lpSum([x[r][3] for (r) in self.R])) / self.D[3]

        # The problem data is written to an .lp file
        prob.writeLP("LPPooblem.lp")

        # The problem is solved using PuLP's choice of Solver
        prob.solve()

        # The status of the solution is printed to the screen
        print("Status:", LpStatus[prob.status])

        # for r in self.R:
        #     for h in self.H:
        #         print ("x" + str(r) + str(h), x[r][h].varValue)
        #
        # # The optimised objective function value is printed to the screen
        # print ("costo total = ", value(prob.objective))

        return x

# Create the 'prob' variable to contain the problem data
# R = list(range(5))
# H = list(range(4))
# RX = list(range(5))
# 
# II = [[0, 0, 0, 0],
#       [2, 1, 1, 3],
#       [0, 0, 2, 5],
#       [3, 3, 2, 1],
#       [0, 3, 5, 1]]
# 
# 
# D = [5,10,15,20]
# 
# A = [6,7,8,9,10]
# 
# M=1000000;
# 
# CF = 100
# CV = 10
# 
# prob = LpProblem("LPOptimizationProblem",LpMinimize)
# 
# x   = pulp.LpVariable.matrix("x", (RX,H),0,None,LpInteger)
# I0  = pulp.LpVariable.dicts("I0", H,0,None,LpInteger)
# F   = pulp.LpVariable.dicts("Fh", H,0,None,LpInteger)
# YI0 = pulp.LpVariable.dicts("YI0h", H,0,1,LpInteger)
# YF  = pulp.LpVariable.dicts("YFh", H,0,1,LpInteger)
# 
# cov = lpSum([I0[h]*CV for (h) in H])
# cof = lpSum([F[h]*CF for (h) in H])
# 
# # The objective function is added to 'prob' first
# prob += cov+cof
# 
# for h in H:
#     prob += -II[1][h] -x[0][h] +D[h]<=M*YI0[h]
#     
# for h in H:
#     prob += II[1][h]+x[0][h]-D[h]<=M*(1-YI0[h])
# 
# for h in H:
#     prob += I0[h]>=0
#     
# for h in H:
#     prob += I0[h]>= II[1][h]+x[0][h]-D[h]
#               
# for h in H:
#     prob += I0[h]<= M*(1-YI0[h])        
#               
# for h in H:
#     prob += I0[h]<= II[1][h]+x[0][h]-D[h]+M*YI0[h]                  
# 
# ########################################
# 
# for h in H:
#     prob += -D[h] + lpSum([II[r][h] for (r) in R]) + lpSum([x[r][h] for (r) in R]) <= M*YF[h]
# 
# for h in H:
#     prob += D[h] - lpSum([II[r][h] for (r) in R]) - lpSum([x[r][h] for (r) in R])<= M*(1-YF[h])
#                                 
# for h in H:
#     prob += F[h]>=0 
#     
# for h in H:
#     prob += F[h]>= D[h] - lpSum([II[r][h] for (r) in R]) - lpSum([x[r][h] for (r) in R]),"Const" + str(h)
#               
# for h in H:
#     prob += F[h]<= M*(1-YF[h])     
#               
# for h in H:
#     prob += F[h]<= D[h] - lpSum([II[r][h] for (r) in R]) - lpSum([x[r][h] for (r) in R]) + M*YF[h]            
# 
# for h in H:
#     prob += F[h]<= 0.25*lpSum([F[h1] for (h1) in H])
# 
# for r in R:
#     prob += A[r] == lpSum([x[r][h] for (h) in H])           
#                           
# 
# 
# prob += (lpSum([II[r][0] for (r) in R])+lpSum([x[r][0] for (r) in R]))/D[0] == (lpSum([II[r][1] for (r) in R])+lpSum([x[r][1] for (r) in R]))/D[1]
# prob += (lpSum([II[r][1] for (r) in R])+lpSum([x[r][1] for (r) in R]))/D[1] == (lpSum([II[r][2] for (r) in R])+lpSum([x[r][2] for (r) in R]))/D[2]
# prob += (lpSum([II[r][2] for (r) in R])+lpSum([x[r][2] for (r) in R]))/D[2] <= (lpSum([II[r][3] for (r) in R])+lpSum([x[r][3] for (r) in R]))/D[3]
# 
# 
# # The problem data is written to an .lp file
# prob.writeLP("LPPooblem.lp")
# 
# # The problem is solved using PuLP's choice of Solver
# prob.solve()
# 
# # The status of the solution is printed to the screen
# print ("Status:", LpStatus[prob.status])
# 
# # Each of the variables is printed with it's resolved optimum value
# 
# for r in R:
#     for h in H:
#         print ("x" + str(r) + str(h), x[r][h].varValue)
#     
# # The optimised objective function value is printed to the screen
# print ("costo total = ", value(prob.objective))

II = [[0, 0, 0, 0],
      [2, 1, 1, 3],
      [0, 0, 2, 5],
      [3, 3, 2, 1],
      [0, 3, 5, 1]]


D = [5,10,15,20]

A = [6,7,8,9,10]

M=1000000;

CF = 100
CV = 10
R=5
H=4

a = AllocationOptimizer(II,A,D,CV,CF,R,H)
print(a.allocate())
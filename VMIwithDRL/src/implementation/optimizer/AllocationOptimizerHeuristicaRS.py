# Import PuLP modeler functions
from collections import namedtuple
import docplex.mp
from docplex.mp.model import Model
from docplex.util.environment import get_environment


# from __future__ import print_function

# import cplex


class AllocationOptimizer():
    def __init__(self, II, A, D, CV, CF, R, H):
        self.II = II
        self.A = A
        self.D = D
        self.CV = CV
        self.CF = CF
        self.M = 1000000
        self.R = list(range(R))
        self.H = list(range(H))
        self.RS = [54, 30, 20, 16]

    # def allocate(self, **kwargs):
    def allocate(self):
        a = [[0 for r in range(len(self.R))] for h in range(len(self.H))]

        II_total = [sum(i) for i in self.II]
        orders = [max(0, self.RS[i] - II_total[i]) for i in range(len(II_total))]
        s = sum(orders)
        if s == 0:
            p = [0.5 / 1.1, 0.3 / 1.1, 0.2 / 1.1, 0.1 / 1.1]
        else:
            p = [i / s for i in orders]
        for r in range(5):
            val1 = int((p[0]) * self.A[r])
            val2 = int((p[1]) * self.A[r])
            val3 = int((p[2]) * self.A[r])
            val4 = int((p[3]) * self.A[r])
            s = val1 + val2 + val3 + val4
            dif = self.A[r] - s
            val1 += dif
            a[0][r] = val1
            a[1][r] = val2
            a[2][r] = val3
            a[3][r] = val4

        #             print(mdl.get_solve_status())
        #             print(a,'\n')
        #             print(self.A)
        #             print(a,'\n')
        return a, False


#
if __name__ == '__main__':
    print("prueba")
    # II = [[0, 2, 0, 3, 0], [0, 1, 0, 3, 3], [0, 1, 2, 2, 5], [0, 3, 5, 1, 1]]
    # D = [5, 5, 5, 5]
    # A = [6, 7, 8, 9, 10]

    II = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    D = [5, 10, 15, 20]
    A = [0, 0, 0, 0, 0]

    M = 1000000
    CF = 100
    CV = 10
    R = 5
    H = 4

    a = AllocationOptimizer(II, A, D, CV, CF, R, H)
    x = a.allocate()

    print(x)

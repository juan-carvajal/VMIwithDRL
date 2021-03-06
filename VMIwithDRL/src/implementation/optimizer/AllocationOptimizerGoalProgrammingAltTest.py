# Import PuLP modeler functions
from collections import namedtuple
import docplex.mp
from docplex.mp.model import Model
from docplex.util.environment import get_environment
import logging


# from _future_ import print_function

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

    # def allocate(self, **kwargs):
    def allocate(self):
        try:
            # self.R = list(range(5))
            # print(self.II)
            RX = list(range(len(self.R)))
            with Model(name='LPOptimizationProblem') as mdl:
                mdl.context.cplex_parameters.threads = 12

                x = mdl.integer_var_matrix(self.H, RX, 0, None, "x")
                I0 = mdl.integer_var_dict(self.H, 0, None, "I0")
                F = mdl.integer_var_dict(self.H, 0, None, "Fh")

                mdl.minimize(mdl.sum(
                    (I0[h] * self.CV + F[h] * self.CF) for h in self.H))

                mdl.add_constraints(I0[h] >= 0 for h in self.H)
                mdl.add_constraints(I0[h] >= self.II[h][0] + x[h, 0] - self.D[h] for h in self.H)
                mdl.add_constraints(F[h] >= 0 for h in self.H)
                mdl.add_constraints(
                    F[h] >= self.D[h] - mdl.sum([self.II[h][r] for (r) in self.R]) - mdl.sum(
                        [x[h, r] for (r) in self.R])
                    for h
                    in self.H)

                mdl.add_constraints(mdl.sum(x[h, r] for h in self.H) == self.A[r] for r in self.R)

                mdl.set_time_limit(60)
                mdl.solve()
                # print("--------------------------------------------------------------------------------------")
                # print(self.II)
                # print(self.D)
                # print(self.A)
                a = [[x[h, r].solution_value for r in range(len(self.R))] for h in range(len(self.H))]
                # print(a)
                #
                # print([I0[h].solution_value for h in range(4)])
                # print([F[h].solution_value for h in range(4)])

            # best_solve_gap=mdl.solve_details.mip_relative_gap
            # if best_solve_gap>0.02:
            #     mdl.set_time_limit(40)
            #     print("Resolving again, Initial Solve Gap:",best_solve_gap)
            #     mdl.solve()
            #     new_solve_gap=mdl.solve_details.mip_relative_gap
            #     if new_solve_gap<best_solve_gap:
            #         best_solve_gap=new_solve_gap
            #         for r in range(5):
            #             for h in range(4):
            #                 a[h][r] = x[h, r].solution_value
            #     print("Final Solve Gap:", best_solve_gap)

            # The status of the solution is printed to the screen

            # print("Status:", mdl.get_solve_status())
            # print(mdl.solve_details)
            # print(mdl._get_solution())
            #     a = [[0 for r in range(len(self.R))] for h in range(len(self.H))]
            #     # mdl.print_solution()
            #     # print(list(x.values()))
            #     # The optimised objective function value is printed to the scree
            #     # print(type(x[0][0]))
            #     # print (a)
            #     for r in range(5):
            #         for h in range(4):
            #             a[h][r] = x[h, r].solution_value
            # print("x" + str(h) + str(r), x[h, r].solution_value)
            #             print(mdl.get_solve_status())
            #             print(a,'\n')
            # print(mdl._get_solution())
            return a, True

        except Exception as e:
            print(e)
            # logging.exception("An exception was thrown!")
            #             for r in range(5):
            #                 for h in range(4):
            #                     share=self.A[r]//4
            #                     remainder=self.A[r]%4
            #                     if h==0:
            #                         a[h][r] = share+remainder
            #                     else:
            #                         a[h][r]=share
            a = [[0 for r in range(len(self.R))] for h in range(len(self.H))]
            for r in range(5):
                val1 = int((0.5 / 1.1) * self.A[r])
                val2 = int((0.3 / 1.1) * self.A[r])
                val3 = int((0.2 / 1.1) * self.A[r])
                val4 = int((0.1 / 1.1) * self.A[r])
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

    II = [[9, 0, 4, 0, 0], [0, 0, 21, 0, 21], [0, 0, 13, 5, 9], [0, 0, 23, 0, 0]]
    D = [40, 26, 30, 18]
    A = [50, 12, 68, 6, 3]

    M = 1000000
    CF = 100
    CV = 5
    R = 5
    H = 4
    a = AllocationOptimizer(II, A, D, CV, CF, R, H)
    x = a.allocate()


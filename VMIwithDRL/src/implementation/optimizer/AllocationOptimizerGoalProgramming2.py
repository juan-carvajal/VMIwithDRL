# Import PuLP modeler functions
from collections import namedtuple
import docplex.mp
from docplex.mp.model import Model
from docplex.util.environment import get_environment
import logging


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

    # def allocate(self, **kwargs):
    def allocate(self):
        try:
            # self.R = list(range(5))
            # print(self.II)
            RX = list(range(len(self.R)))
            mdl = Model(name='LPOptimizationProblem')

            x = mdl.integer_var_matrix(self.H, RX, 0, None, "x")
            I0 = mdl.integer_var_dict(self.H, 0, None, "I0")
            F = mdl.integer_var_dict(self.H, 0, None, "Fh")
            YI0 = mdl.binary_var_dict(self.H)
            YF = mdl.binary_var_dict(self.H)
            cons5_plus = mdl.continuous_var_dict(self.H, 0, None, "Constraint5+")
            cons5_minus = mdl.continuous_var_dict(self.H, 0, None, "Constraint5-")
            cons7_plus = mdl.continuous_var_dict(self.H, 0, None, "Constraint7+")
            cons7_minus = mdl.continuous_var_dict(self.H, 0, None, "Constraint7-")

            coeff = [1, 1, 1, 1]

            mdl.minimize(mdl.sum(
                (coeff[0] * (I0[h] * self.CV + F[h] * self.CF)) + coeff[1] * cons5_plus[h] + coeff[2] * cons7_plus[h] +
                coeff[3] * cons7_minus[h] for h in self.H))

            mdl.add_constraints([- self.II[h][1] - x[h, 1] + self.D[h] <= self.M * YI0[h] for h in self.H])

            mdl.add_constraints(self.II[h][1] + x[h, 1] - self.D[h] <= self.M * (1 - YI0[h]) for h in self.H)

            mdl.add_constraints(I0[h] >= 0 for h in self.H)

            mdl.add_constraints(I0[h] >= self.II[h][1] + x[h, 1] - self.D[h] for h in self.H)

            mdl.add_constraints(I0[h] <= self.M * (1 - YI0[h]) for h in self.H)

            mdl.add_constraints(I0[h] <= self.II[h][1] + x[h, 1] - self.D[h] + self.M * YI0[h] for h in self.H)

            ########################################

            mdl.add_constraints(
                -self.D[h] + mdl.sum([self.II[h][r] for (r) in self.R]) + mdl.sum(
                    [x[h, r] for (r) in self.R]) <= self.M *
                YF[h] for h in self.H)

            mdl.add_constraints(
                self.D[h] - mdl.sum([self.II[h][r] for (r) in self.R]) - mdl.sum(
                    [x[h, r] for (r) in self.R]) <= self.M * (
                        1 - YF[h]) for h in self.H)

            mdl.add_constraints(F[h] >= 0 for h in self.H)

            mdl.add_constraints(
                F[h] >= self.D[h] - mdl.sum([self.II[h][r] for (r) in self.R]) - mdl.sum([x[h, r] for (r) in self.R])
                for h
                in self.H)

            mdl.add_constraints(F[h] <= self.M * (1 - YF[h]) for h in self.H)

            mdl.add_constraints(F[h] <= self.D[h] - mdl.sum([self.II[h][r] for (r) in self.R]) - mdl.sum(
                [x[h, r] for (r) in self.R]) + self.M * YF[h] for h in self.H)

            # Constraint 5

            # mdl.add_constraints(F[h] <= (1/len(self.H)) * mdl.sum([F[h1] for (h1) in self.H]) for h in self.H)
            mdl.add_constraints(
                -cons5_plus[h] + cons5_minus[h] + F[h] - (1 / len(self.H)) * mdl.sum([F[h1] for (h1) in self.H]) == 0
                for h in self.H)

            for r in self.R:
                mdl.add_constraint(self.A[r] == mdl.sum([x[h, r] for (h) in self.H]))

            # Constraint 7
            # for h in self.H[:-1]:
            #     mdl.add_constraint(cons7_plus[h] ==
            #                        (mdl.sum([self.II[h][r] for (r) in self.R]) + mdl.sum([x[h, r] for (r) in self.R])) / self.D[
            #     h] - (
            #                            mdl.sum([self.II[h+1][r] for (r) in self.R]) + mdl.sum([x[h+1, r] for (r) in self.R])) /
            #                    self.D[h+1]
            #
            #
            #
            #                         )

            for h in self.H[:-1]:
                mdl.add_constraint(
                    (mdl.sum([self.II[h][r] for (r) in self.R]) + mdl.sum([x[h, r] for (r) in self.R])) / self.D[
                        h] - (mdl.sum([self.II[h + 1][r] for (r) in self.R]) + mdl.sum(
                        [x[h + 1, r] for (r) in self.R])) /
                    self.D[h + 1] - cons7_plus[h] + cons7_minus[h] == 0
                )
            #

            #             mdl.add_constraint((mdl.sum([self.II[0][r] for (r) in self.R]) + mdl.sum([x[0, r] for (r) in self.R])) / self.D[
            #                 0] == (
            #                                        mdl.sum([self.II[1][r] for (r) in self.R]) + mdl.sum([x[1, r] for (r) in self.R])) /
            #                                self.D[1])
            #             mdl.add_constraint((mdl.sum([self.II[1][r] for (r) in self.R]) + mdl.sum([x[1, r] for (r) in self.R])) / self.D[
            #                 1] == (
            #                                        mdl.sum([self.II[2][r] for (r) in self.R]) + mdl.sum([x[2, r] for (r) in self.R])) /
            #                                self.D[2])
            #             mdl.add_constraint((mdl.sum([self.II[2][r] for (r) in self.R]) + mdl.sum([x[2, r] for (r) in self.R])) / self.D[
            #                 2] <= (
            #                                        mdl.sum([self.II[3][r] for (r) in self.R]) + mdl.sum([x[3, r] for (r) in self.R])) /
            #                                self.D[3])

            mdl.solve()

            # The status of the solution is printed to the screen

            # print("Status:", mdl.get_solve_status())
            # print(mdl.solve_details)
            # print(mdl._get_solution())
            a = [[0 for r in range(len(self.R))] for h in range(len(self.H))]
            # mdl.print_solution()
            # print(list(x.values()))
            # The optimised objective function value is printed to the scree
            # print(type(x[0][0]))
            # print (a)
            for r in range(5):
                for h in range(4):
                    a[h][r] = x[h, r].solution_value
                    # print("x" + str(h) + str(r), x[h, r].solution_value)
            #             print(mdl.get_solve_status())
            #             print(a,'\n')
            return a, True

        except Exception as e:
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

    II = [[27, 13, 24, 5, 0], [15, 5, 11, 1, 0], [10, 3, 7, 1, 0], [0, 0, 2, 0, 0]]
    D = [27, 9, 12, 7]
    A = [0, 0, 37, 28, 0]

    M = 1000000
    CF = 100
    CV = 7.5
    R = 5
    H = 4

    a = AllocationOptimizer(II, A, D, CV, CF, R, H)
    x = a.allocate()

    print(x)

import numpy as np
import math
import pandas as pd


def generateData():
    data = {"d1": [], "d2": [], "d3": [], "d4": [], "donors": []}
    day = 1
    for i in range(365):
        d = get_demand(day)
        data["d1"].append(d[0])
        data["d2"].append(d[1])
        data["d3"].append(d[2])
        data["d4"].append(d[3])
        data["donors"].append(get_donors(day))
        day = (day % 7) + 1
    df = pd.DataFrame(data)
    df.to_csv("run_parameters.csv")


def get_donors(day):
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


def get_demand(day):
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

    d1 = checkDemand(d1)
    d2 = checkDemand(d2)
    d3 = checkDemand(d3)
    d4 = checkDemand(d4)

    demands = [d1, d2, d3, d4]
    # demands=[10,15,8,11]
    return demands


def checkDemand(a):
    a = math.floor(a)
    if (a == 0):
        a = 1
    return a


if __name__ == '__main__':
    generateData()

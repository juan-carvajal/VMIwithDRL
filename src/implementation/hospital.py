class Hospital():

    def __init__(self, ii=None, demand_dist=None, exp_cost=None, stockout_cost=None):
        self.exp_cost = exp_cost
        self.stockout_cost = stockout_cost
        self.inventory = ii;
        self.demand_dist = demand_dist
        self.stockout = []

    def supply(self, supply, demand):
        inventory_aux = [0] * len(self.inventory)

        expired = self.inventory[0]
        stockout = max(0, demand - (sum(self.inventory) + sum(int(v) for v in supply)))

        for i, val in enumerate(self.inventory):
            if i == 0:
                inventory_aux[i] = max(0, self.inventory[i + 1] + supply[i + 1] - demand)
            elif 0 < i < 4:
                inventory_aux[i] = max(0, self.inventory[i + 1] + supply[i]
                                       - max(0, demand - (
                            sum(self.inventory[:i + 1]) + sum(int(v) for v in supply[:i + 1]))))
            else:
                inventory_aux[i] = max(0,
                                       supply[i] - max(0, demand - (
                                                   sum(self.inventory[:i + 1]) + sum(int(v) for v in supply[:i + 1]))))

        self.stockout.append(stockout)
        return (expired * self.exp_cost) + (stockout * self.stockout_cost)

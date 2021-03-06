from implementation.inventory import FIFOInventory


class Hospital():

    def __init__(self, ii, exp_cost, stockout_cost):
        self.exp_cost = exp_cost
        self.stockout_cost = stockout_cost
        #self.inventory = FIFOInventory(ii);
        self.inventory=ii

    def supply(self, supply, demand):

        # self.inventory.push(supply)
        # stockout = self.inventory.pop(demand)
        # expired = self.inventory.age()
        inventory_aux = self.inv_sub_routine(self.inventory, supply, demand)
        expired = max(0, self.inventory[0] + supply[0] - demand)
        stockout = max(0, demand - (sum(self.inventory[1:]) + sum(int(v) for v in supply)))

        for i, val in enumerate(self.inventory):
            if i == 0:
                inventory_aux[i] = max(0, self.inventory[i + 1] + supply[i + 1] - demand)
            elif 0 < i < 4:
                inventory_aux[i] = max(0, self.inventory[i + 1] + supply[i] - max(0, demand - (
                            sum(self.inventory[:i + 1]) + sum(int(v) for v in supply[:i + 1]))))
            else:
                inventory_aux[i] = max(0, supply[i] - max(0, demand - (
                            sum(self.inventory[:i + 1]) + sum(int(v) for v in supply[:i + 1]))))

        self.inventory = inventory_aux
        return ((expired * self.exp_cost) + (stockout * self.stockout_cost)), stockout, expired

    def inv_sub_routine(self, inventory, supply, demand):
        inventory_aux = [0] * len(inventory)
        for i in range(len(inventory_aux)):
            if i == len(inventory_aux) - 1:
                inventory_aux[i] = supply[i]
            else:
                inventory_aux[i] = inventory[i + 1] + supply[i]

        d = demand
        for i in range(len(inventory_aux)):
            if d > 0:
                rest = inventory_aux[i] if d > inventory_aux[i] else d
                inventory_aux[i] -= rest
                d -= rest

        return inventory_aux

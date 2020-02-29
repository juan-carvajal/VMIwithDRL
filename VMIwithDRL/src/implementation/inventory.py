class FIFOInventory:
    def __init__(self, inventory):
        self.inventory = inventory

    def push(self, supply):
        self.inventory = [x + y for x, y in zip(self.inventory, supply)]

    def age(self):
        expired = self.inventory[0]
        aux = [0] * len(self.inventory)
        for i in range(len(aux) - 1):
            aux[i] = self.inventory[i + 1]
        self.inventory = aux
        return expired

    def pop(self, demand):
        remaining = demand
        for i in range(len(self.inventory)):
            if remaining > 0:
                if remaining >= self.inventory[i]:
                    remaining -= self.inventory[i]
                    self.inventory[i] = 0
                else:
                    self.inventory[i] -= remaining
                    remaining = 0
            else:
                break
        return remaining

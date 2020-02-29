from inventory import FIFOInventory


if __name__=='__main__':
    inv=FIFOInventory([10,10,10,10,10])
    print(inv.inventory)
    inv.push([0,0,0,0,1])
    print(inv.inventory)
    exp=inv.age()
    print(exp,inv.inventory)
    st=inv.pop(6544)
    print(st,inv.inventory)
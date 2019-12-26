from implementation.hospital import Hospital
import numpy as np
from implementation.optimizer.AllocationOptimizerHeuristica import AllocationOptimizer
from agent_model.model import Model
#from optimizer.AllocationOptimizerCplexDocPlex import AllocationOptimizer
import timeit
import math
from scipy import stats



class VMI(Model):
    # LOS ESTADOS DE ESTA CLASE SON LOS NIVELES DE INVENTARIOS QUE TIENE PARA CADA UNA DE LAS CADUCIDADES
    # LAS ACCIONES POSIBLES VAN DESDE 0 HASTA MAX_A
    def __init__(self, hospitals, max_A, shelf_life, initial_state=None, exp_cost=None, stockout_cost=None):
        super(VMI, self).__init__(initial_state, max_A, len(initial_state))
        self.day = 1
        self.shelf_life = shelf_life
        self.exp_cost = exp_cost
        self.stockout_cost = stockout_cost
        self.hospitals = [Hospital([0] * shelf_life, None, 1.5*exp_cost, stockout_cost) for _ in range(hospitals)]
        #[Hospital([0] * shelf_life, None, exp_cost*1.5, stockout_cost*1.5)] * hospitals
        
        self.log={}


    def model_logic(self, state, action,options=None):
        #demands = [5, 10, 15, 20]
        donors = self.get_donors(state[5])
        demands = self.get_demand(state[5])
        #donors = self.get_donors()
        #A = min(action,sum(state[:self.shelf_life]))
        A=action
            
        A_i = [0] * self.shelf_life
        for i, val in enumerate(A_i):
            if (i == 0):
                A_i[i] = min(A, state[i])
            else:
                A_i[i] = min(A - sum(A_i[0:i]), state[i])

        II = []
        for i in self.hospitals:
            #print(i.inventory)
            II.append(i.inventory)


        opt = AllocationOptimizer(II, A_i, demands, self.exp_cost, self.stockout_cost, self.shelf_life,
                                  len(self.hospitals))

        # opt = AllocationOptimizer(II, A_i, demands, self.exp_cost, self.stockout_cost, self.shelf_life, len(self.hospitals))
        rep = opt.allocate()

        #print(rep)
        next_state = self.update_inventory_bloodbank(state, donors, action)
        # print(donors)
        # print(next_state)

        reward = state[0] * self.exp_cost
        #reward=0
        #print(reward)
        rewards=[]
        stockouts=[]
        expireds=[]
        
        for hosp in range(len(self.hospitals)):
            r,st,exp=self.hospitals[hosp].supply(rep[hosp], demands[hosp])
            rewards.append(-r)
            stockouts.append(st)
            expireds.append(exp)
            reward += r

        
        if options and options[2]==False:
            year=options[0]
            data={"rewards":rewards , "stockouts":stockouts,"expirees":expireds,"allocation":rep,"action":A,"inventory":state[:self.shelf_life],"donors":donors,"reward":reward,"demands":demands,'DC_expirees':state[0],'II':II}
            if year in self.log:
                self.log[year].append(data)
            else:
                self.log[year]=[]
                self.log[year].append(data)
        # print(reward)
        reward*=-1
        return state, action, next_state, reward, False
    
    
    
    def valid_actions(self, state):
        t_inv=sum(state[:self.shelf_life])+1
        a_max=min(t_inv,self.action_dim)
        v_act=[*range(a_max)]
        #print(v_act)
        return v_act
    
    
    def reset_model(self):
        self.hospitals = [Hospital([0] * self.shelf_life, None, self.exp_cost*1.5, self.stockout_cost) for _ in range(len(self.hospitals))]

    def update_inventory_bloodbank(self, state, donors, action):
        state_aux = [0] * len(state)
        for i in range(self.shelf_life):
            if (i == 0):
                state_aux[i] = max(0, state[i + 1] - action)
            elif 0 < i < 4:
                state_aux[i] = max(0, state[i + 1] - max(0, action - sum(state[:i])))
            elif(i==4):
                state_aux[i] = max(0, donors - max(0, action - sum(state[:i])))
        
        
        state_aux[5] = (state[5]%7) +1         
        #state = state_aux;

        return state_aux

    def get_donors(self,day):
            
            if day == 1:
                don = np.random.triangular(50,90, 120)
            
            elif day ==2:
                don = np.random.triangular(50,90, 120)
                
            elif day ==3:
                don = np.random.triangular(50,90, 120)
            
            elif day ==4:
                don =  np.random.triangular(50,90, 120)    
    
            elif day ==5:
                don = np.random.triangular(50,90, 120)
            
            elif day ==6:
                don = np.random.triangular(50,90, 120)
            else:
                don = np.random.triangular(50,90, 120)
    
            don = math.floor(don)
            #don=100
            return don
    
    
    
    def get_demand(self, day):
        #VENTA DIRECTA UNION TEMPORAL
        
        if day == 1:
            d1 = np.random.gamma(2.6, 6.1)*0.5
            d2 = np.random.gamma(2.6, 6.1)*0.3
            d3 = np.random.gamma(2.6, 6.1)*0.2
            d4 = np.random.gamma(2.6, 6.1)*0.1
        
        elif day ==2:
            d1 = np.random.gamma(4.9, 9.2)*0.5
            d2 = np.random.gamma(4.9, 9.2)*0.3
            d3 = np.random.gamma(4.9, 9.2)*0.2
            d4 = np.random.gamma(4.9, 9.2)*0.1
            
        elif day ==3:
            d1 = np.random.gamma(6.9, 8.2)*0.5
            d2 = np.random.gamma(6.9, 8.2)*0.3
            d3 = np.random.gamma(6.9, 8.2)*0.2
            d4 = np.random.gamma(6.9, 8.2)*0.1
        
        elif day ==4:
            d1 = np.random.gamma(4.7, 9.3)*0.5
            d2 = np.random.gamma(4.7, 9.3)*0.3
            d3 = np.random.gamma(4.7, 9.3)*0.2
            d4 = np.random.gamma(4.7, 9.3)*0.1              

        elif day ==5:
            d1 = np.random.gamma(5.7, 8.0)*0.5
            d2 = np.random.gamma(5.7, 8.0)*0.3
            d3 = np.random.gamma(5.7, 8.0)*0.2
            d4 = np.random.gamma(5.7, 8.0)*0.1 
        
        elif day ==6:
            d1 = np.random.gamma(4.8, 8.7)*0.5
            d2 = np.random.gamma(4.8, 8.7)*0.3
            d3 = np.random.gamma(4.8, 8.7)*0.2
            d4 = np.random.gamma(4.8, 8.7)*0.1     
        else:
            d1 = np.random.gamma(1.7, 3.2)*0.5
            d2 = np.random.gamma(1.7, 3.2)*0.3
            d3 = np.random.gamma(1.7, 3.2)*0.2
            d4 = np.random.gamma(1.7, 3.2)*0.1  
                              
        
        d1 = self.checkDemand(d1)
        d2 = self.checkDemand(d2)
        d3 = self.checkDemand(d3)
        d4 = self.checkDemand(d4)
        
        demands = [d1,d2,d3,d4]
        #demands=[10,15,8,11]
        return demands
    
    def checkDemand(self, a):
        a = math.floor(a)
        if(a == 0):
            a = 1
        return a

        




# agent=QAgent(model,0.99,0.1)
# agent.run(365, 1000)
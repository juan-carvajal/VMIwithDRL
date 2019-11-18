
from numpy import log as ln



def log_epsilon(run,runs):
        return max(0.01, ((0.01 -1) / ln((1-0.25)*runs)) * ln(run+1) + 1      )
    
    
    
    
for i in range(500):
    print(i,log_epsilon(i,500))
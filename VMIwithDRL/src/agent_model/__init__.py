# 
# from numpy import log as ln
# from math import cos
# from math import pi
# 
# 
# def cos_epsilon(runs,run,min_e,min_e_por):
#     #((0.5*COS(H5*PI()/5))+0.5)*(1-(H5/350))
#     c=((((1-min_e)/2.0)*cos(run*pi/5))+(min_e+((1-min_e)/2.0)))*(1-(run/((1.0-min_e_por)*runs)))
#     return max(min_e,c )
#     
#     
#     
#     
# for i in range(500):
#     print(i,cos_epsilon(500,i,0.01,0.25))
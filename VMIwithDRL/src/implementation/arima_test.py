import pmdarima as pm
from numpy.random import randint

if __name__=='__main__':

    data=[65 , 64 , 52 , 48, 59]*5
    automodel = pm.auto_arima(data,
                              start_p=1,
                              start_q=1,
                              test="adf",
                              seasonal=False,
                              trace=False)
    fc=automodel.predict(n_periods=1, return_conf_int=False)
    print(fc[0])
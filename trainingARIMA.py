import pandas as pd
import datetime as dt
import methods as m
import matplotlib.pyplot as plt
import numpy as np
import math
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


def difference(data):
    diff = []
    for i in range(1, len(data)):
        value = data[i] - data[i - 1]
        diff.append(value)
    return np.array(diff)


def predict(coef, history):
    yhat = 0.0
    for i in range(1, len(coef) + 1):
        yhat += coef[i - 1] * history[-i]
    return yhat

def pred(train, test, n_lag,n_ma):

    history = train.tolist()
    model = ARIMA(history, order=(n_lag, 1, n_ma))
    model_fit = model.fit(disp=0,trend='nc')
    ar_coef, ma_coef = model_fit.arparams, model_fit.maparams
    resid = model_fit.resid

    window = train[-n_lag+1:].tolist()
    preds = []
    j = 0

    for t in range(len(test)):
        diff = difference(window)
        if (j < n_ma):
            xf = window[-1] + predict(ar_coef, diff) + predict(ma_coef, [resid[j]])
        else:
            xf = window[-1] + predict(ar_coef, diff)
        preds.append(xf)
        window.append(test[j])
        j += 1

    return predictions



if __name__ == '__main__':

    startTrain = 53

    ### Load in data and normalise
    twitterColumns = [0, 2]
    pollColumns = [1,3, 4, 5, 6, 7, 8,9]  # avdate, Remain (norm), Leave (norm)
    lh, rh, p = m.getPanda(twitterColumns,pollColumns)
    h_agg, p_agg, p_var = m.aggregate(lh, rh, p, splitPolls=False,interpolate=True)

    kalmanData = m.getKalmanData(p_agg, h_agg)

    all_data = kalmanData['remain_perc'].iloc[startTrain:]
    remain_data = all_data.values
    dates_train = all_data.index

    # run experiment for lags and MA
    n_lag = 4
    n_ma = 3
    runs = 100
    k = np.arange(3,4)
    res = np.zeros(shape=(len(k),n_lag-1))
    sorted = np.zeros(shape=(len(k),n_lag-1))


    #print fit
    optimal_type = 'c'
    optimal_lag = 3

    startDate = kalmanData.index[0]+dt.timedelta(days=startTrain+optimal_lag)
    endDate = kalmanData.index[-1]
    pred_dates = pd.date_range(start=startDate, end=endDate)
    end_train = 14

    test = remain_data[-end_train:]
    train = remain_data[:-end_train]

    preds = []
    history = train.tolist()
    for t in range(len(test)):
        model = ARIMA(history, order=(optimal_lag, 0, 2))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0][0]
        preds.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))

    error = mean_squared_error(test, preds)
    print("MSE", math.sqrt(error))
    print(preds)
    predictions = np.concatenate((np.array(model_fit.predict(start=0,end=len(train)-4)), np.array(preds)))

    line1, = plt.plot(dates_train,remain_data, label="Truth", marker='o')
    line2, = plt.plot(pred_dates,predictions, label="Predictions", marker='o')
    plt.axvline(x=dt.datetime(2016, 6, 23), label="Brexit vote", color='r', linestyle='--')
    plt.axvline(x=endDate-dt.timedelta(days=end_train), label="Brexit vote", color='g', linestyle='--')

    handles = [line1, line2,]
    plt.xlabel("Date")
    plt.ylabel("# of tweets")
    plt.legend(handles=handles)
    plt.show()


from pandas import read_csv
import datetime as dt
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot as plt
import methods as m
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import pmdarima as pm

# run a repeated experiment
def experiment_length(series, n_repeats, folds):

    lag = 1
    diff = 1
    ma = 1
    length_error =[]
    data = np.array(np.array_split(series, folds))

    for j in np.arange(1,folds):
        error = []
        for r in range(n_repeats):
            for k in np.arange(j,folds):

                test = data[k]
                if(j>1):
                    train = np.concatenate(data[k-j:k]).ravel()
                else:
                    train = data[k-j:k][0]

                error.append(pred(train, test, lag, diff,ma))

        length_error.append(np.average(error))
        print(length_error)
    return length_error


def pred(train,test,lag,diff,ma):
    # do rolling predictions
    history = [x for x in train]
    predictions = []
    for t in range(len(test)):
        model = ARIMA(history, order=(lag, diff, ma))
        model_fit = model.fit(disp=0,trend='nc')
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)

    error = mean_squared_error(test,predictions)

    return math.sqrt(error)


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

if __name__ == '__main__':

    startTrain = 53
    n_lag = 1
    n_diff = 1
    n_ma = 1

    ### Load in data and normalise
    twitterColumns = [0, 2]
    pollColumns = [1,3, 4, 5, 6, 7, 8,9]  # avdate, Remain (norm), Leave (norm)
    lh, rh, p = m.getPanda(twitterColumns,pollColumns)
    h_agg, p_agg, p_var = m.aggregate(lh, rh, p, splitPolls=False,interpolate=True)

    kalmanData = m.getKalmanData(p_agg, h_agg)

    all_data = kalmanData['remain_perc'].iloc[startTrain:]
    remain_data = all_data.values
    dates_train = all_data.index

    # prepare training and test set
    startDate = kalmanData.index[0] + dt.timedelta(days=startTrain + n_lag+n_diff)
    endDate = kalmanData.index[-1]
    pred_dates = pd.date_range(start=startDate, end=endDate)
    end_train = math.floor(len(remain_data)*0.2)
    predictions = []
    m.setFonts('timeseries')
    test = remain_data[-end_train:]
    train = remain_data[:-end_train]
    history = train.tolist()

    #autofit ARIMA model
    stepwise_model = pm.auto_arima(remain_data, start_p=1, start_q=1,
                           max_p=4, max_q=4,max_order=10,seasonal=False,
                                   d=1,
                                   stationary=False,trace=True,stepwise=True,test='adf',with_intercept=False)

    print(stepwise_model.aic())

    stepwise_model.fit(train)
    print(stepwise_model.summary())


    model = ARIMA(train, order=(1, 1, 1))
    model_fit = model.fit(disp=0, trend='nc')
    ar_coef, ma_coef = model_fit.arparams, model_fit.maparams
    resid = model_fit.resid

    window = remain_data[:2].tolist()
    preds = []
    it = np.arange(startTrain +n_lag+n_diff, len(kalmanData))
    print(it)
    j = 0
    for i in it:
        diff = difference(window)
        if(j<len(resid)):
            xf = window[-1] + predict(ar_coef, diff) + predict(ma_coef, [resid[j]])
        else:
            xf = window[-1] + predict(ar_coef, diff)
        preds.append(xf)
        window.append(remain_data[j+n_lag+n_diff])
        j+=1

    predictions = preds
    error = mean_squared_error(remain_data[-23:], preds[-23:])
    print("MSE", math.sqrt(error))

    print("basic rmse", math.sqrt(mean_squared_error(predictions,remain_data[n_lag+n_diff:])))
    print("test set rmse", math.sqrt(mean_squared_error(predictions[-23:],remain_data[-23:])))
    print("rmse with persistence model", math.sqrt(mean_squared_error(predictions[1:],remain_data[n_lag+n_diff:-1])))

    fig, ax = plt.subplots()
    line1, = ax.plot(dates_train, remain_data, label="Truth", marker='o', color='blue')
    line2, = ax.plot(pred_dates, predictions, label="Predictions", marker='o', color='red')
    plt.axvline(x=dt.datetime(2016, 6, 23), label="Brexit vote", color='slategray', linestyle='-.')
    ax.fill_between([dt.datetime(2016, 6, 23), endDate - dt.timedelta(days=end_train)], 0, 100,
                    facecolor='gainsboro', interpolate=True)

    handles = [line1, line2]
    plt.xlabel("Date")
    plt.ylabel("% support")
    plt.ylim([0, 100])
    plt.legend(handles=handles)
    plt.show()
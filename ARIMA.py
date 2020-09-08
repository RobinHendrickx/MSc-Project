import datetime as dt
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot as plt
import methods as m
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import pmdarima as pm

""" Function that calculates error for increasing length
     Inputs: series - 1D numpy array with timeseries data
             n_repeats - number of times the experiment should be repeated
             folds - number of folds to use during training
     Output: length_error - list of length(# of folds - 1) with errors for length of training set
                            (index 0 is 1 fold, index 2 is 2 folds,...) 
"""
def experiment_length(series, n_repeats, folds):
    lag = 1
    diff = 1
    ma = 1
    length_error = []
    data = np.array(np.array_split(series, folds))

    for k in np.arange(1, folds):
        error = []
        test = data[k]
        train = np.concatenate(data[:k]).ravel()
        for r in range(n_repeats):
            error.append(pred(train, test, lag, diff, ma))
        length_error.append(np.average(error))

    return length_error


""" Function does one out predictions with retraining
     Inputs: train - 1D numpy array with timeseries training data
             test - 1D numpy array with timeseries training data
             lag - integer for number of lags to include in model
             diff - number of differencing steps to use
             ma - number of moving average terms to include
     Output: math.sqrt(error) - mean squared error of test vs predictiosn 
"""
def pred(train, test, lag, diff, ma):
    history = [x for x in train]
    predictions = []
    for t in range(len(test)):
        model = ARIMA(history, order=(lag, diff, ma))
        model_fit = model.fit(disp=0, trend='nc')
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)

    error = mean_squared_error(test, predictions)

    return math.sqrt(error)


""" Function that calculates first order difference of data
    Inputs: data - 1D array-like object with data to be differenced
    Output: np.array(diff) - 1D np array with differenced data
"""
def difference(data):
    diff = []
    for i in range(1, len(data)):
        value = data[i] - data[i - 1]
        diff.append(value)
    return np.array(diff)


""" Function used in one out predictions without retraining
    Inputs: coeff - coefficients to be used (can be MA or AR coefficients)
            history - 1D array-like object with previous terms to include (error residuals in case of MA, 
                      lagged terms in case of AR)
    Output: np.array(diff) - 1D np array with differenced data
"""
def predict(coef, history):
    yhat = 0.0
    for i in range(1, len(coef) + 1):
        yhat += coef[i - 1] * history[-i]
    return yhat



""" MAIN FUNCTION
    This section 1) fits the optimal ARIMA model automatically
                 2) evaluates the fit of the optimal model using one-out predictions without retraining
                 3) calculates the performance of the optimal model when increasing training set size """
if __name__ == '__main__':

    startTrain = 53 # index at which to start training (corresponds to 1st of March with interpolation)
    n_lag = 1       # number of lags to include in ARIMA model
    n_diff = 1      # number of differencing steps
    n_ma = 1        # number of ARIMA terms to include

    ### Load in data and normalise
    twitterColumns = [0, 2]
    pollColumns = [1, 3, 4, 5, 6, 7, 8, 9]
    lh, rh, p = m.getPanda(twitterColumns, pollColumns)
    h_agg, p_agg, p_var = m.aggregate(lh, rh, p, splitPolls=False, interpolate=True)

    kalmanData = m.getKalmanData(p_agg, h_agg) # panda that holds both twitter and polling data

    all_data = kalmanData['remain_perc'].iloc[startTrain:]
    remain_data = all_data.values
    dates_train = all_data.index

    # prepare training and test set
    startDate = kalmanData.index[0] + dt.timedelta(days=startTrain + n_lag + n_diff)
    endDate = kalmanData.index[-1]
    pred_dates = pd.date_range(start=startDate, end=endDate)
    end_train = math.floor(len(remain_data) * 0.2)
    predictions = []
    m.setFonts('timeseries')
    test = remain_data[-end_train:]
    train = remain_data[:-end_train]
    history = train.tolist()

    # 1) autofit ARIMA model
    stepwise_model = pm.auto_arima(remain_data, start_p=1, start_q=1,
                                   max_p=4, max_q=4, max_order=10, seasonal=False,
                                   d=1,
                                   stationary=False, trace=True, stepwise=True, test='adf', with_intercept=False)

    print(stepwise_model.aic())

    stepwise_model.fit(train)
    print(stepwise_model.summary())

    # 2) make one-out predictions for the optimal ARIMA model
    model = ARIMA(train, order=(1, 1, 1))
    model_fit = model.fit(disp=0, trend='nc')
    ar_coef, ma_coef = model_fit.arparams, model_fit.maparams
    resid = model_fit.resid

    window = remain_data[:2].tolist()
    preds = []
    it = np.arange(startTrain + n_lag + n_diff, len(kalmanData))
    j = 0
    for i in it:
        diff = difference(window)
        if (j < len(resid)):
            xf = window[-1] + predict(ar_coef, diff) + predict(ma_coef, [resid[j]])
        else:
            xf = window[-1] + predict(ar_coef, diff)
        preds.append(xf)
        window.append(remain_data[j + n_lag + n_diff])
        j += 1

    # evaluate the error with respect to the persistence model and the truth data
    predictions = preds
    error = mean_squared_error(remain_data[-23:], preds[-23:])
    print("MSE", math.sqrt(error))

    print("basic rmse", math.sqrt(mean_squared_error(predictions, remain_data[n_lag + n_diff:])))
    print("test set rmse", math.sqrt(mean_squared_error(predictions[-23:], remain_data[-23:])))
    print("rmse with persistence model", math.sqrt(mean_squared_error(predictions[1:], remain_data[n_lag + n_diff:-1])))

    # plot results
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

    # 3) calculates the performance of the optimal model when increasing training set size
    # optimal_length = experiment_length(remain_data,1,5)
    # print("error per number of folds as training set = " + str(optimal_length))


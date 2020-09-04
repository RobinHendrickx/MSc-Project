import pandas as pd
import datetime as dt
import methods as m
import matplotlib.pyplot as plt
import numpy as np
import math
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import ar_select_order

# run a repeated experiment
def experiment_lags(series, n_lag, n_repeats, folds):

    data = np.array(np.array_split(series, folds))
    trends = ['c']
    lag_errors = []

    for i in np.arange(1,n_lag):
        errors = []
        for r in range(n_repeats):
            for t in trends:
                for k in np.arange(2,folds):
                    for j in np.arange(0,1):

                        test = data[k]
                        train = np.concatenate(data[j:k]).ravel()
                        if len(train)> i:
                            #fit autoregressive model
                            model = AutoReg(train, lags=i, trend=t)
                            try:
                                model_fit = model.fit()
                            except:
                                print("Couldnt fit model!")

                            predictions = pred(train, test, i, model_fit.params)
                            rmse = math.sqrt(mean_squared_error(test, predictions))
                            errors.append(rmse)
                        else:
                            print("Dataset too small for %f folds" % folds)
        lag_errors.append(np.average(errors))

    return lag_errors


# run a repeated experiment
def experiment_length(series,lag, n_repeats, folds):

    data = np.array(np.array_split(series, folds))
    trends = ['c']
    length_error = []
    for j in np.arange(1,folds):
        error = []
        for r in range(n_repeats):
            for t in trends:
                for k in np.arange(j,folds):

                    test = data[k]
                    if(j>1):
                        train = np.concatenate(data[k-j:k]).ravel()
                    else:
                        train = data[k-j:k][0]

                    #fit autoregressive model
                    model = AutoReg(train, lags=lag, trend=t)
                    try:
                        model_fit = model.fit()
                    except:
                        print("Couldnt fit model!")
                    predictions = pred(train, test, lag, model_fit.params)
                    print(len(train))
                    rmse = math.sqrt(mean_squared_error(test, predictions))
                    error.append(rmse)

        print("training length",len(train))
        length_error.append(np.average(error))

    return length_error


# run a repeated experiment
def experiment_type(series,lag, n_repeats, folds):
    data = np.array(np.array_split(series, folds))
    trends = ['c', 'ct', 't']
    type_error = []

    x = 0
    for t in trends:
        errors = []
        for r in range(n_repeats):
            for k in np.arange(2,folds):
                for j in np.arange(0,k):

                    test = data[k]
                    train = np.concatenate(data[j:k]).ravel()
                    if len(train)> i:
                        #fit autoregressive model
                        model = AutoReg(train, lags=lag, trend=t)
                        try:
                            model_fit = model.fit()
                        except:
                            print("Couldnt fit model!")

                        predictions = pred(train, test, lag, model_fit.params)
                        rmse = math.sqrt(mean_squared_error(test, predictions))
                        errors.append(rmse)
                    else:
                        print("Dataset too small for %f folds" % folds)
        type_error.append(np.average(errors))

    return type_error


def pred(train, test, n_lag,coeff):

    history = train[len(train) - n_lag:]
    history = [history[i] for i in range(len(history))]
    predictions = list()
    for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length - n_lag, length)]
        #yhat = 0
        yhat = coeff[0]
        for d in range(n_lag):
            yhat += coeff[d + 1] * lag[n_lag - d - 1]
            #yhat += coeff[d] * lag[n_lag - d - 1]
        obs = test[t]
        predictions.append(yhat)
        history.append(obs)

    return predictions

if __name__ == '__main__':

    ### SET PARAMETERS ###
    td =  0 # Time delay between twitter and polls
    rel_shift = 0.7
    tPolls = math.floor(td*rel_shift)
    tTwitter = math.ceil((1-rel_shift)*td)

    startTrain = 53
    m.setFonts('timeseries')
    ### Load in data and normalise
    twitterColumns = [0, 2]
    pollColumns = [1,3, 4, 5, 6, 7, 8,9]  # avdate, Remain (norm), Leave (norm)
    lh, rh, p = m.getPanda(twitterColumns,pollColumns)
    h_agg, p_agg, p_var = m.aggregate(lh, rh, p, splitPolls=False,interpolate=True)
    _ , p_onl, p_tel = m.aggregate(lh,rh,p,splitPolls=True,interpolate=True)

    kalmanData = m.getKalmanData(p_agg, h_agg)

    all_data = kalmanData['remain_perc'].iloc[startTrain:]
    remain_data = all_data.values
    dates_train = all_data.index

    # run experiment for lags
    print("order to use",ar_select_order(remain_data, maxlag=13)._aic)
    n_lag = 7
    runs = 100
    k = np.arange(4,5)
    res = np.zeros(shape=(len(k),n_lag-1))
    sorted = np.zeros(shape=(len(k),n_lag-1))

    for i,j in enumerate(k):
        res[i] = experiment_lags(remain_data, n_lag, runs, j)

    print("sum of results: " + str(np.sum(res,axis=0)))
    optimal_lag = np.argmin(np.sum(res,axis=0))+1
    print("optimal lag = " + str(optimal_lag))


    optimal_lag = 1
    runs = 3
    # Now train series for the optimal length (how long do i train to have optimal results?)
    optimal_length = experiment_length(remain_data,optimal_lag,runs,5)
    print("optimal length = " + str(optimal_length))

    #print fit
    optimal_type = 'c'
    optimal_lag = 1

    startDate = kalmanData.index[startTrain]+dt.timedelta(days=optimal_lag)
    endDate = kalmanData.index[-1]
    pred_dates = pd.date_range(start=startDate, end=endDate)
    end_train = math.floor(len(remain_data)*0.2)

    test = remain_data[-end_train:]
    train = remain_data[:-end_train]
    model = AutoReg(train, lags=optimal_lag, trend=optimal_type)
    model_fit = model.fit()
    coeff = model_fit.params
    predictions = np.concatenate((np.array(model.predict(coeff,optimal_lag,len(train)-1)), np.array(pred(train,test,optimal_lag,coeff))))

    print("basic rmse", math.sqrt(mean_squared_error(predictions,remain_data[optimal_lag:])))
    print("test set rmse", math.sqrt(mean_squared_error(predictions[-23:],remain_data[-23:])))
    print("rmse with persistence model", math.sqrt(mean_squared_error(predictions[1:],remain_data[optimal_lag:-1])))


    plt.plot(dates_train[optimal_lag:-1], predictions[1:])
    plt.plot(dates_train[optimal_lag:-1], remain_data[optimal_lag:-1])
    plt.show()

    fig, ax = plt.subplots()
    line1, = ax.plot(dates_train,remain_data, label="Truth", marker='o',color='blue')
    line2, = ax.plot(pred_dates,predictions, label="Predictions", marker='o',color='red')
    plt.axvline(x=dt.datetime(2016, 6, 23), label="Brexit vote", color='slategray', linestyle='-.')
    ax.fill_between([dt.datetime(2016, 6, 23),endDate-dt.timedelta(days=end_train)], 0, 100,
                     facecolor='gainsboro', interpolate=True)

    handles = [line1, line2]
    plt.xlabel("Date")
    plt.ylabel("% support")
    plt.ylim([0,100])
    plt.legend(handles=handles)
    plt.show()
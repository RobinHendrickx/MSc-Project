import pandas as pd
import datetime as dt
import methods as m
import matplotlib.pyplot as plt
import math
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from pandas.plotting import lag_plot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import arch.unitroot as a
import statsmodels.api as sm

if __name__ == '__main__':

    startTrain = 52
    window = 7

    train_percent = 0.8
    test_percent = 0.1
    m.setFonts()

    ### Load in data and normalise
    twitterColumns = [0, 2]
    pollColumns = [1,3, 4, 5, 6, 7, 8,9]  # avdate, Remain (norm), Leave (norm)
    lh, rh, p = m.getPanda(twitterColumns,pollColumns)
    h_agg, p_agg, p_var = m.aggregate(lh, rh, p, splitPolls=False,interpolate=True)

    kalmanData = m.getKalmanData(p_agg, h_agg)
    startDate = kalmanData.index[0]+dt.timedelta(days=startTrain+window)
    endDate = kalmanData.index[-1]

    all_data = kalmanData['remain_perc'].iloc[startTrain:]
    remain_data = all_data.values
    dates_train = all_data.index

    split_train = int(train_percent * len(remain_data))
    split_val = int((1-test_percent)*len(remain_data))

    train = remain_data[:split_train]
    val = remain_data[split_train:split_val]
    test = remain_data[split_val:]

    X = remain_data

    # 1) look at series in isolation
    ############ Autocorrelation ######################

    ########## TEST IF RANDOM WALK ################
    # p value less than 0.05, reject null hypothesis
    # test 1: AD fuller test (null hypothesis: unit root exists)
    print("\nTEST 1: ADF")
    result = adfuller(X, regression='c')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

    # test 2: KPSS test (null hypothesis: unit root does not exist)
    print("\nTEST 2: KPSS")
    result = kpss(X, regression='c', nlags='auto', store=False)
    print('kpss Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('lag parameter: %f' % result[2])
    for key, value in result[3].items():
        print('\t%s: %.3f' % (key, value))

    # test 3@ variance ratio (null hypothesis: random walk, possibly with drift)
    print("\nTEST 3: VARIANCE RATIO")
    print(a.VarianceRatio(X, lags=30, trend='c', debiased=True, robust=True, overlap=True).summary())

    # test 4 DF GLS (null: process contains a unit root)
    dfgls = a.DFGLS(X)
    print(dfgls.summary())

    # test 5
    pp = a.PhillipsPerron(X)
    pp.trend = 'ct'
    print(pp.summary())

    # series in conjunction
    ##########  Correlation between the two with smoothing ##########################

    # Lag plot
    plt.rcParams.update({'ytick.left': False, 'axes.titlepad': 10})

    fig, axes = plt.subplots(2, 4, figsize=(10, 3), sharex=True, sharey=True, dpi=100)
    for i, ax in enumerate(axes.flatten()[:]):
        lag_plot(all_data, lag=i + 1, ax=ax, c='firebrick')
        ax.set_title('Lag ' + str(i + 1))

    fig.suptitle(
        'Lag Plots\n',y=1.15)
    plt.show()

    pd.plotting.autocorrelation_plot(all_data)

    sm.graphics.tsa.plot_pacf(all_data.values.squeeze(), lags=8)
    plt.show()

    # 2) fit autoregressive model
    model = AutoReg(X, window, trend='c', seasonal=False, exog=None, hold_back=None, period=None,
                                     missing='none')
    model_fit = model.fit()
    print('coefficients: %s' % model_fit.params)

    coef = model_fit.params

    # walk forward over time steps in test
    history = train[len(train) - window:]
    history = [history[i] for i in range(len(history))]
    predictions = list()
    for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length - window, length)]
        yhat = coef[0]
        for d in range(window):
            yhat += coef[d + 1] * lag[window - d - 1]
        obs = test[t]
        predictions.append(yhat)
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    rmse = math.sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)

    # plot
    plt.plot(test)
    plt.plot(predictions, color='red')
    plt.show()





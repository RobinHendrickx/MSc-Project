import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import methods as m
from statsmodels.nonparametric.smoothers_lowess import lowess
import math
from pandas.plotting import lag_plot
from sklearn.metrics import mean_squared_error
from math import sqrt

# Helper function that calculates optimisation value. X and Y are the two timeseries to be calculated for
# descr says which value to optimise by. options: "mse", "corr"
def optimise(X,Y,descr):
    if descr == "mse":
        return -sqrt(mean_squared_error(X, Y))

    if descr == "corr":
        return np.corrcoef(np.reshape(X,(1,-1)), np.reshape(Y,(1,-1)))[0][1]


#Fit the twitter data X (df1[c1]) and polling data Y (df2[c2]) for optimisation
def fitLoessRaw(X,Y,cx,cy):
    max = -10000

    res = []
    params = {}
    timeshift = np.arange(0, 17)
    A = np.arange(0.45, 0.8, 0.05)
    A = [1]
    b = np.arange(10, 35, 1)
    b = [0]
    Y_loess_5 = pd.DataFrame(lowess(Y[cy], np.arange(len(Y[cy])), frac=0)[:, 1],
                              index=Y.index, columns=[cy])
    X_loess_5 = pd.DataFrame(lowess(X[cx], np.arange(len(X[cx])), frac=0.15)[:, 1],
                             index=X[cx].index, columns=[cx])

    for j in timeshift:
        for i in A:
            for k in b:
                y = Y_loess_5.copy()
                x = X_loess_5.copy()

                if int(j) != 0:
                    x.index += dt.timedelta(days=int(j))

                x = x * i + k
                Z = pd.merge(x, y, left_index=True, right_index=True)
                x = Z[cx].values
                y = Z[cy].values
                value = optimise(np.reshape(x,(1,-1)),np.reshape(y,(1,-1)),"corr")
                length_set = x.size
                res.append(value)

                if value > max and length_set >= 10:
                    max = value
                    params = {"opt_val": value, "A": i, "B": k, "timeshift": j, "length": length_set,"corr":optimise(np.reshape(x,(1,-1)),np.reshape(y,(1,-1)),"corr"),"rmse":optimise(np.reshape(x,(1,-1)),np.reshape(y,(1,-1)),"mse")}

    return params


# MAIN
if __name__ == '__main__':

    ### GET DATA ###
    m.setFonts('timeseries')
    startDate = dt.datetime(year=2016, month=3, day=1)
    endDate = dt.datetime(year=2016, month=6, day=1)
    interpolate = False
    m.longPrint()
    ### Load in data and normalise
    twitterColumns = [0, 2]
    pollColumns = [1,3, 4, 5, 6, 7, 8, 9]  # avdate, Remain (norm), Leave (norm)
    lh, rh, p = m.getPanda(twitterColumns, pollColumns)
    h_agg, p_agg, p_var = m.aggregate(lh, rh, p, splitPolls=False, interpolate=interpolate)
    _, p_onl, p_tel = m.aggregate(lh, rh, p, splitPolls=True, interpolate=interpolate)

    kalmanData = m.getKalmanData(p_agg, h_agg)
    kalmanData_o = m.getKalmanData(p_onl,h_agg)
    kalmanData_t = m.getKalmanData(p_tel,h_agg)

    # 1. Moving Average
    df_orig = kalmanData
    df_ma = df_orig.rolling(3, center=True, closed='both').mean()

    # 2. Loess Smoothing (5% and 15%)
    print("LOESS SMOETHING")
    df_loess_5 = pd.DataFrame(lowess(df_orig, np.arange(len(df_orig)), frac=0.05)[:, 1],
                              index=df_orig.index, columns=['remain_perc'])
    df_loess_15 = pd.DataFrame(lowess(df_orig, np.arange(len(df_orig)), frac=0.15)[:, 1],
                               index=df_orig.index, columns=['remain_perc'])

    # Plot loess smoothed
    fig, axes = plt.subplots(4, 1, figsize=(7, 7), sharex=True, dpi=120)
    df_orig.plot(ax=axes[0], color='k', title='Original Series')
    df_loess_5.plot(ax=axes[1], title='Loess Smoothed 5%')
    df_loess_15.plot(ax=axes[2], title='Loess Smoothed 15%')
    df_ma.plot(ax=axes[3], title='Moving Average (3)')
    fig.suptitle('Smoothened timeseries', y=0.95, fontsize=14)
    plt.show()


    line1, = plt.plot(kalmanData['Remain'], linestyle='-', color='C1', label='polls')
    line2, = plt.plot(df_loess_5, linestyle='-', color='C2', label='5%')
    line3, = plt.plot(df_loess_15,linestyle='-',color = 'C0', label = '15%')
    line4, = plt.plot(df_ma,color = 'C9',label = "moving average")
    handles = [line1,line2, line3, line4]
    plt.xlabel("Date")
    plt.ylabel("Support in %")
    plt.legend(handles=handles)
    plt.axvline(x=dt.datetime(2016, 6, 23), label="Brexit vote", color='r', linestyle='--')
    plt.show()
    # use smoothing to fit to timeseries?

    fig, axes = plt.subplots(2, 4, figsize=(10, 3), sharex=True, sharey=True, dpi=100)
    for i, ax in enumerate(axes.flatten()[:]):
        lag_plot(df_loess_5, lag=i + 1, ax=ax, c='firebrick')
        ax.set_title('Lag ' + str(i + 1))

    fig.suptitle(
        'Lag Plots for 5% smoothed series\n', y=1.15)
    plt.show()


    # LOESS smoothing fit
    result =  fitLoessRaw(X=h_agg.loc[startDate:endDate], Y=p_agg.loc[startDate:endDate], cx='remain_perc', cy='Remain')
    print("loess fit: ", result)
    result =  fitLoessRaw(X=h_agg.loc[startDate:endDate], Y=p_agg.loc[startDate:endDate], cx='leave_perc', cy='Leave')
    print("loess fit:" , result)

    # plot results
    Y_loess_5 = pd.DataFrame(lowess(p_agg['Remain'], np.arange(len(p_agg['Remain'])), frac=0)[:, 1],
                              index=p_agg.index, columns=['Remain'])
    X_loess_5 = pd.DataFrame(lowess(h_agg['remain_perc'].loc[startDate:endDate], np.arange(len(h_agg['remain_perc'].loc[startDate:endDate])), frac=0.15)[:, 1],
                             index=h_agg.loc[startDate:endDate].index, columns=['remain_perc'])

    x = X_loess_5 * result["A"] + result["B"]
    x.index += dt.timedelta(days=int(result["timeshift"]))
    y = Y_loess_5


    line1, = plt.plot(y.loc[startDate+ dt.timedelta(days=int(result["timeshift"])):endDate+ dt.timedelta(days=int(result["timeshift"]))].index,
                      y.loc[startDate+ dt.timedelta(days=int(result["timeshift"])):endDate+ dt.timedelta(days=int(result["timeshift"]))], linestyle='-', color='blue', label='polls')
    line2, = plt.plot(x.loc[startDate:endDate+ dt.timedelta(days=int(result["timeshift"]))].index,
                      x.loc[startDate:endDate+ dt.timedelta(days=int(result["timeshift"]))], linestyle='-', color='red', label='Tweets')
    handles = [line1, line2]
    plt.xlabel("Date")
    plt.ylabel("Support in %")
    plt.ylim([35, 75])
    plt.legend(handles=handles)
    plt.show()

    Y_loess_5 = pd.DataFrame(lowess(p_agg['Leave'], np.arange(len(p_agg['Leave'])), frac=0)[:, 1],
                             index=p_agg.index, columns=['Leave'])
    X_loess_5 = pd.DataFrame(lowess(h_agg['leave_perc'].loc[startDate:endDate],
                                    np.arange(len(h_agg['leave_perc'].loc[startDate:endDate])), frac=0.15)[:,
                             1],
                             index=h_agg.loc[startDate:endDate].index, columns=['leave_perc'])

    x = X_loess_5 * result["A"] + result["B"]
    x.index += dt.timedelta(days=int(result["timeshift"]))
    y = Y_loess_5


    line1, = plt.plot(y.loc[startDate + dt.timedelta(days=int(result["timeshift"])):endDate + dt.timedelta(
        days=int(result["timeshift"]))].index,
                      y.loc[startDate + dt.timedelta(days=int(result["timeshift"])):endDate + dt.timedelta(
                          days=int(result["timeshift"]))], linestyle='-', color='blue', label='polls')
    line2, = plt.plot(x.loc[startDate:endDate + dt.timedelta(days=int(result["timeshift"]))].index,
                      x.loc[startDate:endDate + dt.timedelta(days=int(result["timeshift"]))], linestyle='-',
                      color='red', label='Tweets')
    handles = [line1, line2]
    plt.xlabel("Date")
    plt.ylabel("Support in %")
    plt.ylim([35, 75])
    plt.legend(handles=handles)
    plt.show()

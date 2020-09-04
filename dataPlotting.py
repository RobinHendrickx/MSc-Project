import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import methods as m


# when normalised data is given, plot both polling and twitter data. Otherwise only twitter.
def plottwitter(df, title):
    cols = ("number_of_tweets_remain", "number_of_tweets_leave")
    line1, = plt.plot(df[cols[0]], label="Remain",color='blue')
    line2, = plt.plot(df[cols[1]], label="Leave",color='red')
    plt.yscale('log')
    plt.axvline(x=dt.datetime(2016, 6, 23), label="Brexit vote", color='slategray', linestyle='-.')

    handles = [line1, line2, ]
    plt.xlabel("Date")
    plt.ylabel("# of tweets")
    # plt.title(title)
    plt.legend(handles=handles)
    plt.show()


# when normalised data is given, plot both polling and twitter data. Otherwise only twitter.
def plotpolls(df, raw=False):
    if (raw):
        line1, = plt.plot(df['Remain_raw'], label="Remain",color='blue')
        line2, = plt.plot(df['Leave_raw'], label="Leave",color='red')
        line3, = plt.plot(df['WNV or DK'], label="Undecided",color='grey')


        handles = [line1, line2, line3]
    else:
        line1, = plt.plot(df['Remain'], label="Remain", color='blue')
        line2, = plt.plot(df['Leave'], label="Leave", color='red')
        handles = [line1, line2]


    plt.axvline(x=dt.datetime(2016, 6, 23), label="Brexit vote", color='slategray', linestyle='-.')

    plt.xlabel("Date")
    plt.ylabel("Support in %")
    plt.legend(handles=handles)
    plt.show()


def plottwitternorm(df, title):
    col_list = ["number_of_tweets_remain", "number_of_tweets_leave"]
    cutoff = 500
    x = df[df[col_list].sum(axis=1) > cutoff]

    cols = ("remain_perc", "leave_perc")

    line1, = plt.plot(df[cols[0]], label="Support Remain", color='blue',linestyle='-')
    line2, = plt.plot(df[cols[1]], label="Support Leave", color='red',linestyle='-')
    plt.xlim([kalmanData.index[0],endDate+dt.timedelta(days=5)])
    plt.axvline(x=dt.datetime(2016, 6, 23), label="Brexit vote", color='slategray', linestyle='-.')

    plt.xlabel("Date")
    plt.ylabel("Support in %")
    handles = [line1, line2,]

    plt.legend(handles=handles)
    plt.show()

    fig, ax = plt.subplots()
    line1, = ax.plot(x[cols[0]], label="Support Remain", color='blue', linestyle='-')
    line2, = ax.plot(x[cols[1]], label="Support Leave", color='red', linestyle='-')
    plt.axvline(x=dt.datetime(2016, 6, 23), label="Brexit vote", color='slategray', linestyle='-.')
    ax.set_xlabel("Date")
    plt.title("more than 500")
    ax.set_ylabel("Support in %")
    plt.xlim([kalmanData.index[0],endDate+dt.timedelta(days=5)])


    handles = [line1, line2]
    ax.legend(handles=handles)
    plt.show()


# MAIN
def plotRollingPolls(p, j):
    Y1 = p['Remain'].rolling(j, center=None).mean()
    Y2 = p['Leave'].rolling(j, center=None).mean()
    line3, = plt.plot(Y1, linestyle='--', alpha=0.4,
                      label="Remain polled")
    line4, = plt.plot(Y2, linestyle='--', alpha=0.4,
                      label="Leave polled")
    handles = [line3, line4]
    plt.title("Rolling average with j = " + str(j))
    plt.legend(handles=handles)
    plt.show()


if __name__ == '__main__':
    m.setFonts('timeseries')

    startTrain = 52
    look_back = 0

    train_percent = 0.8
    test_percent = 0.20
    accumulate = False
    load = False

    ### Load in data and normalise
    twitterColumns = [0, 2]
    pollColumns = [1, 3, 4, 5, 6, 7, 8, 9]  # avdate, Remain (norm), Leave (norm)
    lh, rh, p = m.getPanda(twitterColumns, pollColumns)
    h_agg, p_agg, p_var = m.aggregate(lh, rh, p, splitPolls=False, interpolate=True)

    kalmanData = m.getKalmanData(p_agg, h_agg)
    kalmanData.to_csv("kalmandata.csv")
    startDate = kalmanData.index[0] + dt.timedelta(days=startTrain + look_back)
    endDate = kalmanData.index[-1]

    ### PLOTS ###
    plottwitter(h_agg,"Twitter data")
    plottwitternorm(h_agg,"Twitter data normalised by day")
    plotpolls(p_agg.loc[startDate:endDate],False)
    plotpolls(p)
    plotRollingPolls(p,7)


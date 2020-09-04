import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#global constants
m_startDate = '2016-01-09'
m_startDateTel = '2016-01-24'
m_endDate = '2016-06-23 23:00:00'


def getPanda(twitterColumns,pollColumns):
    # read in data
    lh = readPanda(r'data\support_leave_hour.csv', twitterColumns)
    rh = readPanda(r'data\support_remain_hour.csv', twitterColumns)
    p = readPanda(r'data\polls.csv', pollColumns, False)
    return lh, rh, p


def readPanda(loc, usecols, twitter=True):
    if twitter:
        dateParse = lambda dates: dt.datetime.strptime(dates,
                                                       '%Y-%m-%d %H:%M:%S')
        date = "startdate"
    else:
        dateParse = lambda dates: dt.datetime.strptime(dates, '%Y-%m-%d')
        date = "enddate"

    data = pd.read_csv(loc,
                       parse_dates={'timeline': [date]},
                       date_parser=dateParse,
                       usecols=usecols)

    data = data.sort_values(by="timeline")
    data.set_index('timeline', inplace=True)

    if twitter:
        i = pd.date_range(start=m_startDate, end=m_endDate,
                          freq='H')  # hardcoded to correspond with end of polls
        data = data.reindex(i)

    return data


# Aggregate polling data (ie take average from polls on same day) and aggregate twitter data daily
def aggregate(lh, rh, p, interpolate=False, splitPolls=False):
    hour = pd.merge(lh, rh, left_index=True, right_index=True, suffixes=("_leave", "_remain"))

    ## aggregate per date
    if splitPolls:
        p_onl = p.loc[p['Type'] == "O"]  #note that type column dissappears after taking the mean
        p_tel = p.loc[p['Type'] == "T"]
        p_onl_agg = p_onl.groupby(p_onl.index.date).mean()
        p_tel_agg = p_tel.groupby(p_tel.index.date).mean()
        p_onl_agg.index = pd.to_datetime(p_onl_agg.index)  # make the index a datetime again. bug in the groupby function
        p_tel_agg.index = pd.to_datetime(p_tel_agg.index)  # make the index a datetime again. bug in the groupby function

    else:
        pagg = p
        pagg = pagg.groupby(pagg.index.date).mean()
        pvar = p.groupby(p.index.date).var()
        pagg.index = pd.to_datetime(pagg.index)  # make the index a datetime again. bug in the groupby function

    hour = hour.groupby(hour.index.date).sum(min_count=1) # min_count handles NaNs
    hour.index = pd.to_datetime(hour.index)

    #print(pd.date_range(start=startDate, end=endDate).difference(hour.index))

    # reindex and interpolate to have a continuous range
    if interpolate:
        i_hour = pd.date_range(start=m_startDate, end=m_endDate,
                               freq='D')  # hardcoded to correspond with end of polls
        hour = hour.reindex(i_hour)
        hour = hour.interpolate(method='linear')
        if splitPolls:
            i_polls_o = pd.date_range(start=m_startDate, end=m_endDate,
                                    freq='D')  # hardcoded to correspond with end of polls
            i_polls_t = pd.date_range(start=m_startDateTel, end=m_endDate,
                                    freq='D')  # hardcoded to correspond with end of polls
            p_onl_agg = p_onl_agg.reindex(i_polls_o)
            p_tel_agg = p_tel_agg.reindex(i_polls_t)
            p_tel_agg = p_tel_agg.interpolate(method='linear')
            p_onl_agg = p_onl_agg.interpolate(method='linear')
        else:
            i_polls = pd.date_range(start=m_startDate, end=m_endDate,
                                    freq='D')  # hardcoded to correspond with end of polls
            pagg = pagg.reindex(i_polls)
            pagg = pagg.interpolate(method='linear')
    # calc percentage of supporters for Twitter
    hour['remain_perc'] = np.nan
    hour['leave_perc'] = np.nan
    hour = calcperc(hour)


    if splitPolls:
        return hour, p_onl_agg, p_tel_agg

    return hour, pagg, pvar


# Joins polling and twitter data (kalmanData, joined with h_agg aggregated between polling dates)
def getKalmanData(p_agg, h_agg):

    range = pd.DatetimeIndex(p_agg.index[p_agg.index>=h_agg.index[0]]) # assumes that polls start before twitter and polls finish sooner
    h_ad = h_agg.loc[range[0]:range[-1]] # make sure start and end date match
    custom = h_ad.groupby(range[range.searchsorted(h_ad.index)]).sum()  # ie aggregate between dates of polls
    custom = calcperc(custom)  # important! Cant just sum the percentages
    kalmanData = custom.merge(p_agg, left_index=True, right_index=True)

    return kalmanData


# Helper function that calculates support % from twitter data (ie pre joining)
def calcperc(hour):
    col_list = list(hour)
    col_list.remove('remain_perc')
    col_list.remove('leave_perc')
    hour['remain_perc'] = hour['number_of_tweets_remain'] / hour[col_list].sum(axis=1) * 100.
    hour['leave_perc'] = hour['number_of_tweets_leave'] / hour[col_list].sum(axis=1) * 100.
    hour['remain_perc'] = hour['remain_perc'].fillna(50)
    hour['leave_perc'] = hour['leave_perc'].fillna(50)

    return hour


# Choose fonts for graphs
def setFonts(fig,x=True):
    if not x:
        return

    SMALL_SIZE = 8
    MEDIUM_SIZE = 13
    BIGGER_SIZE = 20

    if fig == 'timeseries':
        plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    else:
        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# Set option to fully print pandas or not
def longPrint(x=True):
    if not x:
        return

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

# shift polls backwards in time. td is the shift in days (positive) and p_agg the daily aggregated polls.
# fake polls are added at the back
def shift_polls(p_agg, td, addFake=False):
    if (td > 0):

        # shift polls
        #try:
        #    p_agg_temp = p_agg.shift(-td, freq='D')
        #except:
        p_agg.index -= dt.timedelta(days=td)
        p_agg_temp = p_agg

        if addFake is True:
            # add fake polls
            p_fake = pd.DataFrame(columns=["Remain", "Leave"], index=pd.date_range(p_agg.index[-1]+dt.timedelta(days=1), periods=td))
            mask = p_agg.index > p_agg.index[-1] - dt.timedelta(days=10)
            mean_r = p_agg['Remain'].loc[mask].mean()  # (shift is not correct! last td values is not the same as last td days!!, it is after interpolation)
            mean_l = p_agg['Leave'].loc[mask].mean()

            for i in range(0, td):
                p_fake['Remain'].iloc[i] = mean_r
                p_fake['Leave'].iloc[i] = mean_l

            # join fake and shifted polls
            p_agg = pd.concat([p_agg_temp, p_fake]).astype('float')
        else:
            p_agg = p_agg_temp
    return p_agg

# shift tweets forward in time
def shift_tweets(h_agg, td):
    #if td > 0:
    #    h_agg = h_agg.shift(td, freq='D')
    h_agg.index += dt.timedelta(days=td)

    return h_agg

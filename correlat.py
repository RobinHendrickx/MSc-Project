import pandas as pd
import datetime as dt
import numpy as np
import methods as m
import math


def correlation(kalmanData,td=0):
    print("\nFrom " + str(kalmanData.index.values[-td])[:10] + " to " + str(kalmanData.index.values[-1])[:10])
    print("corr coeff leave: ", np.corrcoef(kalmanData["number_of_tweets_leave"].iloc[-td:], kalmanData["Leave"].iloc[-td:]))
    print("corr coeff remain: ", np.corrcoef(kalmanData["number_of_tweets_remain"].iloc[-td:], kalmanData["Remain"].iloc[-td:]))

# MAIN
if __name__ == '__main__':

    ### GET DATA ###
    td =  0 # Time delay between twitter and polls
    rel_shift = 0.1
    tPolls = math.floor(td * rel_shift)
    tTwitter = math.ceil((1 - rel_shift) * td)

    interpolate = False

    ### Load in data and normalise
    twitterColumns = [0, 2]
    pollColumns = [1, 3, 4, 5, 6, 7, 8, 9]  # avdate, Remain (norm), Leave (norm)
    lh, rh, p = m.getPanda(twitterColumns, pollColumns)
    h_agg, p_agg, p_var = m.aggregate(lh, rh, p, splitPolls=False, interpolate=interpolate)
    _, p_onl, p_tel = m.aggregate(lh, rh, p, splitPolls=True, interpolate=interpolate)

    p_agg = m.shift_polls(p_agg, tPolls,addFake=False)
    p_onl = m.shift_polls(p_onl,tPolls,addFake=False)
    p_tel = m.shift_polls(p_tel,tPolls,addFake=False)
    h_agg = m.shift_tweets(h_agg, tTwitter)

    ### CORRELATION AND FITTING ###
    #only do non telephone and online split
    startDate = dt.datetime(year=2016,month=3,day=1)
    endDate = dt.datetime(year=2016,month=6,day=23)

    h = h_agg['remain_perc'].loc[startDate:endDate]
    p = p_agg['Remain'].loc[startDate:endDate]
    X = pd.merge(h,p,left_index=True,right_index=True)
    print("Interpolated = "+str(interpolate) + "; all data: ", np.corrcoef(X['remain_perc'],X['Remain']))

    #split in telephone and online
    pt = p_tel['Remain'].loc[startDate.date():endDate.date()]
    po = p_onl['Remain'].loc[startDate.date():endDate.date()]
    Xo = pd.merge(h, po, left_index=True, right_index=True)
    Xt = pd.merge(h, pt, left_index=True, right_index=True)
    print("Interpolated = "+str(interpolate) +"; telephone: ", np.corrcoef(Xt['remain_perc'], Xt['Remain']))
    print("Interpolated = "+str(interpolate) + "; online: ", np.corrcoef(Xo['remain_perc'], Xo['Remain']))



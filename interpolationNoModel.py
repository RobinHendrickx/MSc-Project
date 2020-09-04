import pandas as pd
import datetime as dt
import methods as m
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

if __name__ == '__main__':

    tPolls = 2
    tTwitter = 14
    startTrain = 52
    addFake = False
    m.setFonts('timeseries')
    ### Load in data and normalise
    twitterColumns = [0, 2]
    pollColumns = [1,3, 4, 5, 6, 7, 8,9]  # avdate, Remain (norm), Leave (norm)
    lh, rh, p = m.getPanda(twitterColumns,pollColumns)
    h_agg, p_agg, p_var = m.aggregate(lh, rh, p, splitPolls=False,interpolate=True)
    p_orig = p_agg.copy()
    h_orig = h_agg.copy()
    p_agg = m.shift_polls(p_agg, tPolls,addFake=addFake)
    h_agg = m.shift_tweets(h_agg, tTwitter)

    kalmanData = m.getKalmanData(p_agg, h_agg)
    startDate = kalmanData.index[0]+dt.timedelta(days=startTrain)
    endDate = dt.datetime(day=23, month=6, year=2016)




    ### FIND KF VARIABLES: 1) R and 2) P0
    # find R
    preds = []
    R_r = p_var['Remain'].mean()
    r = kalmanData['remain_perc'].to_numpy(dtype=float)
    P0_r = r.var()/10
    H = 1
    K_r = P0_r * H / (H * P0_r * H + R_r)
    K_r = 0.95

    ### KF MODEL ###
    # apply interpolation
    it = np.arange(startTrain, len(kalmanData))
    kalmanData['result_remain'] = np.nan
    kalmanData['result_leave'] = np.nan

    for i in it:
        preds.append(kalmanData['remain_perc'].iloc[i])
        kalmanData['result_remain'].iloc[i] = kalmanData['remain_perc'].iloc[i] + K_r * (
                    kalmanData['Remain'].iloc[i] - H * kalmanData['remain_perc'].iloc[i])
        kalmanData['result_leave'].iloc[i] = kalmanData['leave_perc'].iloc[i] + K_r * (
                    kalmanData['Leave'].iloc[i] - H * kalmanData['leave_perc'].iloc[i])

    # predict x days ahead
    preds_extra_remain = []
    preds_extra_leave = []

    j = it[-1]+1
    for i in range(tPolls + 5):
        xf = h_agg['remain_perc'].iloc[j]
        preds_extra_remain.append(xf)
        j+=1

    for i in range(len(preds_extra_remain)):
        preds_extra_leave.append(100 - preds_extra_remain[i])

    datelist = pd.date_range(startDate, periods=len(kalmanData)-startTrain + len(preds_extra_remain)).tolist()

    for i in range(len(kalmanData['result_leave'])):
        kalmanData['result_leave'].iloc[i] = 100 - kalmanData['result_remain'].iloc[i]

    all_results_remain = np.concatenate((kalmanData['result_remain'].loc[startDate:].values,
                                         np.array(preds_extra_remain)))  # create lsit for all remain results
    all_results_leave = np.concatenate((kalmanData['result_leave'].loc[startDate:].values,
                                        np.array(preds_extra_leave)))  # create lsit for all remain results

    # print model parameters
    print("\nKalman filter values")
    print("K_r", K_r)
    print("H", H)
    print("R_r", R_r)
    print("P0_r", P0_r)

    if True:
        # PLOT DATA ASSIMILATION RESULTS
        fig, ax = plt.subplots()
        line1, = plt.plot(datelist, all_results_remain, linestyle='--', color='darkslategray', label='Assimilated')
        line2, = plt.plot(h_orig['remain_perc'].loc[startDate:endDate], linestyle='-', color='lightblue', label='Tweets',
                          alpha=0.5)
        line3, = plt.plot(p_orig['Remain'].loc[startDate:'2016-06-23'], linestyle='-', color='lightcoral',
                          label='Polls',
                          alpha=0.5)
        ax.fill_between([dt.datetime(2016, 6, 21), endDate + dt.timedelta(days=5)], 0, 100,
                        facecolor='gainsboro', interpolate=True)
        handles = [line1, line2, line3]
        plt.xlabel("Date")
        plt.ylabel("Support in %")
        plt.legend(handles=handles)
        plt.axvline(x=dt.datetime(2016, 6, 23), label="Brexit vote", color='slategray', linestyle='-.')
        plt.ylim([0, 100])
        plt.show()

        fig, ax = plt.subplots()
        line1, = plt.plot(datelist, all_results_leave, linestyle='--', color='darkslategray', label='Assimilated')
        line2, = plt.plot(h_agg['leave_perc'].loc[startDate:endDate], linestyle='-', color='lightblue', label='Tweets',
                          alpha=0.5)
        line3, = plt.plot(p_orig['Leave'].loc[startDate:'2016-06-23'], linestyle='-', color='lightcoral', label='Polls',
                          alpha=0.5)
        ax.fill_between([dt.datetime(2016, 6, 21), endDate + dt.timedelta(days=5)], 0, 100,
                        facecolor='gainsboro', interpolate=True)
        handles = [line1, line2, line3]
        plt.xlabel("Date")
        plt.ylabel("Support in %")
        plt.legend(handles=handles)
        plt.axvline(x=dt.datetime(2016, 6, 23), label="Brexit vote", color='slategray', linestyle='-.')
        plt.ylim([0, 100])
        plt.show()

    ### VALIDATION ###
    # 1) see if residuals are zero mean gaussian (MAKE SURE TO SET DELAY TO ZERO!)
    preds = np.array(preds).ravel()
    res_l = 100 * np.ones(len(preds)) - preds - kalmanData['Leave'].loc[
                                                str(startDate):str(endDate)].to_numpy()  # all polls
    res_r = preds - kalmanData['Remain'].loc[str(startDate):str(endDate)].to_numpy()  # all polls
    (mu_l, sigma_l) = norm.fit(res_l)
    (mu_r, sigma_r) = norm.fit(res_r)
    m.longPrint()

    # the residual error plot (histogram + normal distr)
    n, bins, patches = plt.hist(res_l, density=True, facecolor='C0', alpha=0.75)
    y = norm.pdf(bins, loc=mu_l, scale=sigma_l)
    plt.plot(bins, y, 'r--', linewidth=2)
    plt.title("Leave pdf with $\mu=" + str(round(mu_l, 2)) + "$ and $\sigma =" + str(round(sigma_l, 2)) + "$")
    plt.show()

    n, bins, patches = plt.hist(res_r, density=True, facecolor='C0', alpha=0.75)
    y = norm.pdf(bins, loc=mu_r, scale=sigma_r)
    plt.plot(bins, y, 'r--', linewidth=2)
    plt.title("Remain pdf with $\mu=" + str(round(mu_r, 2)) + "$ and $\sigma =" + str(round(sigma_r, 2)) + "$")
    plt.show()

    # 2) MSE
    X = kalmanData['result_leave'].loc[startDate:endDate].to_numpy()
    Y = p_orig['Leave'].loc[startDate:endDate - dt.timedelta(days=tPolls)].to_numpy()
    mse_l = sum((X - Y) ** 2) / X.size

    print("mse for leave polls", mse_l)

    X = kalmanData['result_remain'].loc[startDate:endDate].to_numpy()
    Y = h_agg['remain_perc'].loc[startDate:endDate - dt.timedelta(days=tPolls)].to_numpy()
    mse_r = sum((X - Y) ** 2) / X.size
    print("mse for remain shifted tweets", mse_r)


    # the residual error plot (histogram + normal distr)
    n, bins, patches = plt.hist(res_l, density=True, facecolor='C0', alpha=0.75)
    y = norm.pdf(bins, loc=mu_l, scale=sigma_l)
    plt.plot(bins, y, 'r--', linewidth=2)
    plt.title("Leave pdf with $\mu=" + str(round(mu_l, 2)) + "$ and $\sigma =" + str(round(sigma_l, 2)) + "$")
    plt.show()

    n, bins, patches = plt.hist(res_r, density=True, facecolor='C0', alpha=0.75)
    y = norm.pdf(bins, loc=mu_r, scale=sigma_r)
    plt.plot(bins, y, 'r--', linewidth=2)
    plt.title("Remain pdf with $\mu=" + str(round(mu_r, 2)) + "$ and $\sigma =" + str(round(sigma_r, 2)) + "$")
    plt.show()
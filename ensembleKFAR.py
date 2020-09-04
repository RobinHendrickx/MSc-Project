import pandas as pd
import datetime as dt
import methods as m
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from statsmodels.tsa.ar_model import AutoReg

if __name__ == '__main__':

    ### SET PARAMETERS ###
    tPolls = 2
    tTwitter = 14

    startTrain = 53
    n_points = 1000
    n_lag = 1
    trend = 'c'
    addFake = False

    m.setFonts("timeseries")
    ### Load in data and normalise
    twitterColumns = [0, 2]
    pollColumns = [1, 3, 4, 5, 6, 7, 8, 9]  # avdate, Remain (norm), Leave (norm)
    lh, rh, p = m.getPanda(twitterColumns, pollColumns)
    h_agg, p_agg, p_var = m.aggregate(lh, rh, p, splitPolls=False, interpolate=True)
    _, p_onl, p_tel = m.aggregate(lh, rh, p, splitPolls=True, interpolate=True)

    p_orig = p_agg.copy()
    h_orig = h_agg.copy()
    p_agg = m.shift_polls(p_agg, tPolls, addFake=addFake)
    h_agg = m.shift_tweets(h_agg, tTwitter)

    kalmanData = m.getKalmanData(p_agg, h_agg)
    kalmanData.to_csv("kalmandata.csv")
    startDate = kalmanData.index[0] + dt.timedelta(days=startTrain + n_lag)
    endDate = dt.datetime(day=23, month=6, year=2016)

    all_data = kalmanData['remain_perc'].iloc[startTrain:]
    remain_data = all_data.values

    ### FIND KF VARIABLES: 1) R and 2) P0
    # find R
    R_r = p_var['Remain'].mean()
    r = kalmanData['remain_perc'].to_numpy(dtype=float)
    P0_r = r.var() / 10
    R_r = 13.08*10
    P0_r = 266
    x0 = 50
    H = 1
    Q_r = R_r

    # fit autoreg
    model = AutoReg(remain_data, lags=n_lag, trend=trend)
    model_fit = model.fit()
    seq = remain_data[:n_lag]
    coeff = model_fit.params

    ### KF MODEL ###
    it = np.arange(startTrain + n_lag, len(kalmanData))

    kalmanData['result_remain'] = np.nan
    kalmanData['result_leave'] = np.nan


    kalmanData['result_remain'] = np.nan
    kalmanData['result_leave'] = np.nan

    sigmas_seq = np.empty((n_points, n_lag, 1))

    P0_r = P0_r
    sigmas = np.random.normal(0, P0_r, size=(n_points, 1))  # draw from distribution, normally P0_r
    for i in range(n_points):
        sigmas_seq[i] = seq + sigmas[i]

    Ks = []
    preds = []
    xf = np.empty([n_points, 1])
    # assimilation scheme
    for j in it:
        # predict
        for i in range(n_points):
            xf[i] = coeff[0]
            for d in range(n_lag):
                xf[i] += coeff[d + 1] * sigmas_seq[i][n_lag - d - 1]

        xf += np.random.normal(0, Q_r, size=(n_points, 1))  # should be x and p0?
        xmean = np.mean(xf)
        preds.append(xmean)
        # update
        xf_z = H * xf
        zmean = np.mean(xf_z)
        Pzz = np.var(xf_z, ddof=1) + R_r
        Pxz = 0
        for i in range(n_points):
            Pxz += (xf[i] - xmean) * (xf_z[i] - zmean)
        Pxz /= n_points - 1

        K_r = Pxz / Pzz
        Ks.append(K_r)
        v_r = np.random.normal(0, R_r, n_points)  # should be R_r
        y = kalmanData['Remain'].iloc[j]

        for i in range(n_points):
            xf[i] += K_r * (y + v_r[i] - xf_z[i])

        x0 = np.mean(xf)
        P0_r = P0_r - K_r * Pzz * K_r
        kalmanData['result_remain'].iloc[j] = x0

        # create new sequence
        for i in range(n_points):
            seq = np.append(sigmas_seq[i], xf[i])
            sigmas_seq[i] = seq[1:]

        # predict x days ahead
    preds_extra_remain = []
    preds_extra_leave = []

    for _ in range(tPolls + 5):
        # predict
        for i in range(n_points):
            xf[i] = coeff[0]
            for d in range(n_lag):
                xf[i] += coeff[d + 1] * sigmas_seq[i][n_lag - d - 1]

        xf += np.random.normal(0, Q_r, size=(n_points, 1))  # should be x and p0?
        xmean = np.mean(xf)
        preds_extra_remain.append(xmean)
        # create new sequence
        for i in range(n_points):
            seq = np.append(sigmas_seq[i], xf[i])
            sigmas_seq[i] = seq[1:]

    for i in range(len(preds_extra_remain)):
        preds_extra_leave.append(100 - preds_extra_remain[i])
        datelist = pd.date_range(startDate, periods=len(all_data) - n_lag + len(preds_extra_remain)).tolist()

    for i in range(len(kalmanData['result_leave'])):
        kalmanData['result_leave'].iloc[i] = 100 - kalmanData['result_remain'].iloc[i]

    # print model parameters
    print("\nKalman filter values")
    print("K_r", K_r)
    print("H", H)
    print("R_r", R_r)
    print("P0_r", P0_r)

    all_results_remain = np.concatenate((kalmanData['result_remain'].loc[startDate:].values,
                                         np.array(preds_extra_remain)))  # create lsit for all remain results
    all_results_leave = np.concatenate((kalmanData['result_leave'].loc[startDate:].values,
                                        np.array(preds_extra_leave)))  # create lsit for all remain results

    # Plot results of data assimilation
    if True:
        fig, ax = plt.subplots()
        line1, = plt.plot(datelist, all_results_remain, linestyle='--', color='darkslategray', label='Assimilated')

        line2, = plt.plot(h_agg['remain_perc'].loc[startDate:endDate], linestyle='-', color='lightblue', label='Tweets',
                          alpha=0.5)
        line3, = plt.plot(p_orig['Remain'].loc[startDate:'2016-06-23'], linestyle='-', color='lightcoral',
                          label='Polls', alpha=0.5)
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
        line2, = plt.plot(h_orig['leave_perc'].loc[startDate:endDate], linestyle='-', color='lightblue', label='Tweets',
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
    if True == True:
        # 1) see if residuals are zero mean gaussian (MAKE SURE TO SET DELAY TO ZERO!)
        preds = np.array(preds).ravel()
        res_l = 100 * np.ones(len(preds)) - preds - kalmanData['Leave'].loc[
                                                    str(startDate):str(endDate)].to_numpy()  # all polls
        res_r = preds - kalmanData['Remain'].loc[str(startDate):str(endDate)].to_numpy()  # all polls
        (mu_l, sigma_l) = norm.fit(res_l)
        (mu_r, sigma_r) = norm.fit(res_r)

        # 2) MSE
        X = kalmanData['result_leave'].loc[startDate + dt.timedelta(days=n_lag):endDate].to_numpy()
        Y = p_orig['Leave'].loc[startDate + dt.timedelta(days=n_lag):endDate - dt.timedelta(days=tPolls)].to_numpy()
        mse_l = sum((X - Y) ** 2) / X.size

        print("mse for leave", mse_l)

        X = kalmanData['result_remain'].loc[startDate + dt.timedelta(days=n_lag):endDate].to_numpy()
        Y = p_orig['Remain'].loc[startDate + dt.timedelta(days=n_lag):endDate - dt.timedelta(days=tPolls)].to_numpy()
        mse_r = sum((X - Y) ** 2) / X.size
        print("mse for remain", mse_r)


        X = kalmanData['result_leave'].loc[startDate+dt.timedelta(days=n_lag):endDate].to_numpy()
        Y = h_agg['leave_perc'].loc[startDate+dt.timedelta(days=n_lag):endDate- dt.timedelta(days=tPolls)].to_numpy()
        mse_r = sum((X - Y) ** 2) / X.size
        print("mse for leave Twitter",mse_r)

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

    plt.plot(pd.date_range(start=startDate, end=endDate - dt.timedelta(days=tPolls), freq='D'), Ks)
    plt.ylim([0, 1])
    plt.show()

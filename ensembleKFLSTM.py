import pandas as pd
import datetime as dt
import methods as m
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':

    ### SET PARAMETERS ###
    td =  16 # Time delay between twitter and polls
    tPolls = 2
    tTwitter = 14

    startTrain = 52
    length = 3
    n_points = 1000
    addFake = False
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
    kalmanData.to_csv("kalmandata.csv")
    startDate = kalmanData.index[0]+dt.timedelta(days=startTrain+length)
    endDate = dt.datetime(day=23, month=6, year=2016)


    all_data = kalmanData['remain_perc'].iloc[startTrain:]
    remain_data = all_data.values
    remain_data = remain_data.reshape((-1, 1))
    dates_train = all_data.index

    scaler = MinMaxScaler(feature_range=(0, 1))
    remain_data = scaler.fit_transform(remain_data)

    # load model
    json_file = open('models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("models/model.h5")
    print("loaded model from disk")
    model.compile(loss='mse', optimizer='adam')

    ### FIND KF VARIABLES: 1) R and 2) P0
    # find R
    R_r = p_var['Remain'].mean()*10
    r = kalmanData['remain_perc'].to_numpy(dtype=float)
    P0_r = r.var()
    #P0_r = 0.1
    P0_r = 266
    R_r = 13.08*10
    x0 = 50
    H = 1
    Q_r = R_r
    m.setFonts("timeseries")

    ### KF MODEL ###
    it = np.arange(startTrain+length, len(kalmanData))

    kalmanData['result_remain'] = np.nan
    kalmanData['result_leave'] = np.nan
    seq = remain_data[:length]
    seq = np.array(scaler.inverse_transform(seq))
    sigmas_seq = np.empty((n_points,length,1))
    xf = np.empty(n_points)

    sigmas = np.random.normal(0, P0_r, size=(n_points, 1))  # draw from distribution, normally P0_r
    for i in range(n_points):
        sigmas_seq[i] = seq+sigmas[i]
        sigmas_seq[i] = scaler.transform(sigmas_seq[i])

    sigmas_seq = np.reshape(sigmas_seq,(n_points, length,1))

    Ks = []
    preds = []
    # assimilation scheme
    for j in it:
        #predict
        xf = model.predict(sigmas_seq)
        xf = scaler.inverse_transform(xf)
        xf += np.random.normal(0, Q_r, size=(n_points,1)) # should be x and p0?
        xmean = np.mean(xf)
        preds.append(xmean)
        #update
        xf_z = H*xf
        zmean = np.mean(xf_z)
        Pzz = np.var(xf_z,ddof=1)+R_r
        Pxz = 0
        for i in range(n_points):
            Pxz += (xf[i]-xmean)*(xf_z[i]-zmean)
        Pxz/=n_points-1

        K_r = Pxz/Pzz
        Ks.append(K_r)
        v_r = np.random.normal(0,R_r,n_points) #should be R_r
        y = kalmanData['Remain'].iloc[j]

        for i in range(n_points):
            xf[i] += K_r*(y+v_r[i]-xf_z[i])

        x0 = np.mean(xf)
        P0_r = P0_r - K_r*Pzz*K_r
        kalmanData['result_remain'].iloc[j] = x0

        #create new sequence
        xf_scaled = scaler.transform(np.reshape(xf,(n_points,-1)))
        for i in range(n_points):
            seq = np.append(sigmas_seq[i], xf_scaled[i])
            sigmas_seq[i] = np.reshape(seq[1:],(length,1))

        sigmas_seq = np.reshape(sigmas_seq,(n_points,length,1))

    # print model parameters
    print("\nKalman filter values")
    print("K_r", K_r)
    print("H", H)
    print("R_r", R_r)
    print("P0_r", P0_r)

    # predict x days ahead
    preds_extra_remain = []
    preds_extra_leave = []

    for _ in range(tPolls + 5):
        # predict
        xf = model.predict(sigmas_seq)
        xf = scaler.inverse_transform(xf)
        xf += np.random.normal(0, Q_r, size=(n_points, 1))  # should be x and p0?
        x0 = np.mean(xf)
        preds_extra_remain.append(x0)
        xf_scaled = scaler.transform(np.reshape(xf,(n_points,-1)))
        # create new sequence
        for i in range(n_points):
            seq = np.append(sigmas_seq[i], xf_scaled[i])
            sigmas_seq[i] = np.reshape(seq[1:], (length, 1))

        sigmas_seq = np.reshape(sigmas_seq, (n_points, length, 1))


    for i in range(len(preds_extra_remain)):
        preds_extra_leave.append(100 - preds_extra_remain[i])

    datelist = pd.date_range(startDate, periods=len(all_data) - length + len(preds_extra_remain)).tolist()

    for i in range(len(kalmanData['result_leave'])):
        kalmanData['result_leave'].iloc[i] = 100 - kalmanData['result_remain'].iloc[i]


    for i in range(len(kalmanData['result_leave'])):
        kalmanData['result_leave'].iloc[i] = 100-kalmanData['result_remain'].iloc[i]

    # print model parameters
    print("\nKalman filter values")
    print("K_r", K_r)
    print("H", H)
    print("R_r", R_r)
    print("P0_r", P0_r)

    all_results_remain = np.concatenate((kalmanData['result_remain'].loc[startDate:].values,np.array(preds_extra_remain))) # create lsit for all remain results
    all_results_leave = np.concatenate((kalmanData['result_leave'].loc[startDate:].values,np.array(preds_extra_leave))) # create lsit for all remain results
    #h_agg.index -= dt.timedelta(days=tTwitter)

    # Plot results of data assimilation
    if True:
        fig, ax = plt.subplots()
        line1, = plt.plot(datelist,all_results_remain, linestyle='--', color='darkslategray', label='Assimilated')

        line2, = plt.plot(h_orig['remain_perc'].loc[startDate:endDate], linestyle='-', color='lightblue', label='Tweets',alpha=0.5)
        line3, = plt.plot(p_orig['Remain'].loc[startDate:'2016-06-23'], linestyle='-', color='lightcoral', label='Polls',alpha=0.5)
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
    if True == True:
        # 1) see if residuals are zero mean gaussian (MAKE SURE TO SET DELAY TO ZERO!)
        preds = np.array(preds).ravel()
        res_l = 100*np.ones(len(preds))-preds-kalmanData['Leave'].loc[str(startDate):str(endDate)].to_numpy() # all polls
        res_r = preds - kalmanData['Remain'].loc[str(startDate):str(endDate)].to_numpy() # all polls
        (mu_l, sigma_l) = norm.fit(res_l)
        (mu_r, sigma_r) = norm.fit(res_r)


        # 2) MSE
        X = kalmanData['result_leave'].loc[startDate+dt.timedelta(days=length):endDate].to_numpy()
        Y = p_orig['Leave'].loc[startDate+dt.timedelta(days=length):endDate- dt.timedelta(days=tPolls)].to_numpy()
        mse_l = sum((X - Y) ** 2) / X.size

        print("mse for leave polls",mse_l)

        X = kalmanData['result_leave'].loc[startDate+dt.timedelta(days=length):endDate].to_numpy()
        Y = h_agg['leave_perc'].loc[startDate+dt.timedelta(days=length):endDate- dt.timedelta(days=tPolls)].to_numpy()
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


    plt.plot(pd.date_range(start=startDate,end=endDate- dt.timedelta(days=tPolls),freq='D'),Ks)
    plt.ylim([0,1])
    plt.show()
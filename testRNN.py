import pandas as pd
import datetime as dt
import methods as m
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import math
import tensorflow
from keras.models import model_from_json
import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

def train(data):

    model = Sequential()
    model.add(
        LSTM(n_hidden,
             input_shape=(look_back, 1),
             dropout=0.1)
    )
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    data = scaler.fit_transform(data)
    train_generator = keras.preprocessing.sequence.TimeseriesGenerator(data,data, length=look_back,
                                                                       batch_size=batch_size)

    model.fit_generator(train_generator, epochs=num_epochs, verbose=0, shuffle=False)  # train model

    return model

def experiment_length(series, n_repeats, folds):

    lag = 1
    diff = 1
    ma = 1
    length_error =[]
    data = np.array(np.array_split(series, folds))

    for j in np.arange(1,folds):
        error = []
        for r in range(n_repeats):
            for k in np.arange(j,folds):

                test = data[k]
                if(j>1):
                    train = np.concatenate(data[k-j:k]).ravel()
                else:
                    train = data[k-j:k][0]

                error.append(pred(train, test))

        length_error.append(np.average(error))
        print(length_error)
    return length_error


def pred(tr,test):
    model = train(tr)
    it = np.arange(0, len(test))
    remain_train = scaler.transform(tr)
    seq = remain_train[:look_back]
    seq = np.reshape(seq, (1, look_back, 1))
    preds = []

    for i in it:
        xf = model.predict(seq) #predict
        preds.append(xf)
        seq = seq.flatten()
        seq = np.append(seq,scaler.transform(test[i]))
        seq = seq[1:]
        seq = np.reshape(seq,(1,look_back,1))

    preds = scaler.inverse_transform(np.reshape(preds, (1, -1)))
    rmse = math.sqrt(mean_squared_error(remain_test, preds[0,-len(remain_test):]))
    return rmse

if __name__ == '__main__':

    startTrain = 52
    look_back = 3
    num_epochs = 1500
    batch_size = 20
    n_hidden = 2

    train_percent = 0.8
    test_percent = 0.20
    accumulate = False
    load = True

    ### Load in data and normalise
    twitterColumns = [0, 2]
    pollColumns = [1,3, 4, 5, 6, 7, 8,9]  # avdate, Remain (norm), Leave (norm)
    lh, rh, p = m.getPanda(twitterColumns,pollColumns)
    h_agg, p_agg, p_var = m.aggregate(lh, rh, p, splitPolls=False,interpolate=True)

    kalmanData = m.getKalmanData(p_agg, h_agg)
    kalmanData.to_csv("kalmandata.csv")
    startDate = kalmanData.index[0]+dt.timedelta(days=startTrain+look_back)
    endDate = kalmanData.index[-1]

    all_data = kalmanData['remain_perc'].iloc[startTrain:]
    remain_data = all_data.values
    remain_data = remain_data.reshape((-1, 1))
    dates_train = all_data.index

    split_train = int(train_percent * len(remain_data))

    remain_train = remain_data[:split_train]
    remain_test = remain_data[split_train:]

    scaler = MinMaxScaler(feature_range=(0, 1))


    # load model
    if load:
        json_file = open('models/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("models/model.h5")
        print("loaded model from disk")
        model.compile(loss='mse', optimizer='adam')
        print(model.summary())
        scaler.fit_transform(remain_train)
    else:
        model = train(remain_train)
        print(model.summary())

    # testing on all
    it = np.arange(startTrain+look_back, len(kalmanData))
    remain_train = scaler.transform(remain_train)
    seq = remain_train[:look_back]
    seq = np.reshape(seq, (1, look_back, 1))
    preds = []

    for i in it:
        xf = model.predict(seq) #predict
        preds.append(xf)

        #create new sequence
        if accumulate:
            seq = seq.flatten()
            seq = np.append(seq, xf)
            seq = seq[1:]
            seq = np.reshape(seq,(1,look_back,1))
        else:
            seq = seq.flatten()
            seq = np.append(seq,scaler.transform(np.reshape(kalmanData['remain_perc'].iloc[i],(1,-1))))
            seq = seq[1:]
            seq = np.reshape(seq,(1,look_back,1))


    preds = scaler.inverse_transform(np.reshape(preds, (1, -1)))
    rmse = math.sqrt(mean_squared_error(remain_test, preds[0,-len(remain_test):]))
    print("rmse", rmse)

    print("basic rmse", math.sqrt(mean_squared_error(preds[0,:], remain_data[look_back:])))
    print("test set rmse", math.sqrt(mean_squared_error(preds[0,-23:], remain_data[-23:])))
    print("rmse with persistence model",
          math.sqrt(mean_squared_error(preds[0,1:], remain_data[look_back:-1])))

    end_train = math.floor(len(remain_data) * 0.2)
    m.setFonts('timeseries')

    fig, ax = plt.subplots()
    line1, = ax.plot(dates_train, remain_data, label="Truth", marker='o', color='blue')
    line2, = ax.plot(np.reshape(kalmanData.loc[startDate:endDate].index,(-1)),np.reshape(preds,(-1)), label="Predictions", marker='o', color='red')
    plt.axvline(x=dt.datetime(2016, 6, 23), label="Brexit vote", color='slategray', linestyle='-.')
    ax.fill_between([dt.datetime(2016, 6, 23), endDate - dt.timedelta(days=end_train)], 0, 100,
                    facecolor='gainsboro', interpolate=True)

    handles = [line1, line2]
    plt.xlabel("Date")
    plt.ylabel("% support")
    plt.ylim([0, 100])
    plt.legend(handles=handles)
    plt.show()



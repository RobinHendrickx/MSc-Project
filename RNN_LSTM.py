import numpy as np
import pandas as pd
from tensorflow import keras
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense
import methods as m
from sklearn.preprocessing import MinMaxScaler
from keras.regularizers import L1L2
import csv


def run(remain_train, folds, batch_size, look_back, n_hidden,
        num_epochs, layers, dropout, rec_dropout, regul_b, regul_k, regul_r):

    data = np.array(np.array_split(remain_train, folds))

    model = Sequential()
    model.add(
        LSTM(n_hidden,
             input_shape=(look_back, 1),
             dropout=dropout,
             recurrent_dropout=rec_dropout,
             bias_regularizer=regul_b,
             recurrent_regularizer=regul_r,
             kernel_regularizer=regul_k)
    )
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())
    errors = []

    for k in range(1,folds):
        for j in range(0,k):
            test = data[k]
            train = np.reshape(np.concatenate(data[j:k]).ravel(),(-1,1))
            train_generator = keras.preprocessing.sequence.TimeseriesGenerator(train,train, length=look_back,
                                                                               batch_size=batch_size)
            val_generator = keras.preprocessing.sequence.TimeseriesGenerator(test,test, length=look_back,
                                                                                 batch_size=1)
            for i in range(3):
                model.fit_generator(train_generator, epochs=num_epochs, verbose=0, shuffle=False)  # train model
                errors.append(model.evaluate_generator(val_generator))

    # saving the model
    if True:
        model_json = model.to_json()
        with open("models/model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("models/model.h5")
        print("saved model to disk")

    return np.average(errors)






if __name__ == '__main__':

    # model parameters
    startTrain = 52 # where training set starts
    rolling = False # Rolling window yes or no
    rolling_size = 4 # size of rolling window
    train_percent = 1 # percentage split between training and test
    test_percent = 0

    RANDOM_SEED = 30
    np.random.seed(RANDOM_SEED)


    ### PREPROCESS ###
    twitterColumns = [0, 2]  # startdate, # of tweets
    pollColumns = [1,3, 4, 5, 6, 7, 8]  # avdate, Remain (norm), Leave (norm)

    lh, rh, p = m.getPanda(twitterColumns, pollColumns)
    h_agg, p_agg, p_var = m.aggregate(lh, rh, p, splitPolls=False, interpolate=True)
    kalmanData = m.getKalmanData(p_agg, h_agg)

    all_data = kalmanData['remain_perc'].iloc[startTrain:]

    if rolling == True:
        all_data = all_data.rolling(rolling_size, center=None,min_periods=1).mean()

    remain_data = all_data.values
    remain_data = remain_data.reshape((-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    remain_data = scaler.fit_transform(remain_data)


    fields = ['number','batch size','look back', 'n_hidden','n_epochs','layers','rec_dropout','dropout','results']
    with open(r'results.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow(fields)

    # specify hyper parameter files
    batch_size = [20] # size of batch
    look_back = [3]
    n_hidden = np.arange(1,10)
    num_epochs = [1500]
    layers = [1,]
    rec_dropout = [0]
    dropout = [0]
    n_hidden =[2]


    reg_b = [L1L2(l1=0.0,l2=0.0)] # to bias weights
    reg_r = [L1L2(l1=0.0,l2=0.0)] # recurrent weight reguliser
    reg_k = [L1L2(l1=0.0,l2=0.0)] # to inputs

    total = len(batch_size)*len(look_back)*len(n_hidden)*len(num_epochs)*len(layers)*len(rec_dropout) * len(dropout)\
            *len(reg_b)*len(reg_r)*len(reg_k)
    counter = 0


    for a in batch_size:
        for b in look_back:
            for c in n_hidden:
                for d in num_epochs:
                    for e in layers:
                        for f in rec_dropout:
                            for g in dropout:
                                for h in reg_b:
                                    for i in reg_r:
                                        for j in reg_k:

                                            results = run(remain_data, 1,
                                                          a,
                                                          look_back=b,
                                                          n_hidden=c,
                                                          num_epochs=d,
                                                          layers=e,
                                                          rec_dropout=f,
                                                          dropout=g,
                                                          regul_b=h,
                                                          regul_r=i,
                                                          regul_k=j
                                                          )
                                            counter+=1
                                            fields = [str(counter)+"/"+str(total),str(a),str(b),str(c),str(d),str(e),str(f),str(g),results]
                                            with open(r'results.csv', 'a') as file:
                                                writer = csv.writer(file)
                                                writer.writerow(fields)


    #evaluate
    #         epochs     (500,1000,2000,3000,4000,5000)
    #         neurons    linspace (5:500)
    #         layers     (1, 2, 3)
    #         timestep   (7, 14, 21)
    #         batch size (1,20, all)
    #         smoothing  (2,3,4,5,6,7)
    #         dropout    (recurrent, normal)
    #         regularisation (recurrent weight, other)
    #         starting point
    #         sliding or expanding window https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/


    # https://towardsdatascience.com/time-series-forecasting-with-recurrent-neural-networks-74674e289816 OG SOURCE
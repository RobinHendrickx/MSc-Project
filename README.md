This project contains the files for the MSc project "Correcting public opinion trends through machine learning and data assimilation". The version of Python required is 3.6.8.
A brief outline of the project files is provided here:
    
   Data assimilation models
   =======================
          Architecture one
          ----------------
          1) AR models - Assimilation files: interpolationAR.py; ensembleKFAR.py
                         Exploratory work: AnalysisAR.py
                         Model optimisation: trainingAR.py
  
          2) ARIMA models - Assimilation files: interpolationARIMA.py; ensembleKFARIMA.py
                            Model optimisation: ARIMA.py
           
          3) LSTM models - Assimilation files: interpolationLSTM.py; ensembleKFLSTM.py
                           Exploratory work: testRNN.py
                           Model optimisation: RNN_LSTM.py
           
           Architecture two
           ----------------
           1) No model - Assimilation files: interpolationNoModel.py
           
   Other files
   ===========
       1) time lag estimation: curveFitting.py
       2) data analysis: correlat.py; dataPlotting.py; entropy.py
       3) saved RNN models: models/ folder
       4) overarching methods used throughout project: methods.py
       5) data files: data/ folder

  Important sources
  =================
        1) Approximate entropy code snippet - https://gist.github.com/DustinAlandzes/a835909ffd15b9927820d175a48dee41
        2) Ensemble Kalman filter inspiration for Python - https://filterpy.readthedocs.io/en/latest/kalman/EnsembleKalmanFilter.html
        3) Out-of-sample prediction for AR and ARIMA model code snippet https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
                                                                        https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/

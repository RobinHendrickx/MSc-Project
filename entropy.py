import methods as m
import datetime as dt
import numpy as np

# This code snippet was mostly taken from https://gist.github.com/DustinAlandzes/a835909ffd15b9927820d175a48dee41
if __name__ == '__main__':

    ### GET DATA ###
    startTrain = 52
    startDate = dt.datetime(year=2016, month=3, day=1)
    endDate = dt.datetime(year=2016, month=6, day=23)
    interpolate = False

    ### Load in data and normalise
    twitterColumns = [0, 2]
    pollColumns = [1,3, 4, 5, 6, 7, 8, 9]  # avdate, Remain (norm), Leave (norm)
    lh, rh, p = m.getPanda(twitterColumns, pollColumns)
    h_agg, p_agg, p_var = m.aggregate(lh, rh, p, splitPolls=False, interpolate=interpolate)

    kalmanData = m.getKalmanData(p_agg, h_agg)
    remain_data = kalmanData['remain_perc'].loc[startDate:endDate].values

    def ApEn(U, m, r):
        """Compute Aproximate entropy"""
        def _maxdist(x_i, x_j):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

        def _phi(m):
            x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
            C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
            return (N - m + 1.0)**(-1) * sum(np.log(C))

        N = len(U)
        return abs(_phi(m+1) - _phi(m))

    small = []
    big = []

    for i in range(1000):
        rand_small = np.random.randint(1, 100, size=36)
        rand_big = np.random.randint(1, 100, size=len(remain_data))
        small.append(ApEn(rand_small, m=2, r=0.2*np.std(rand_small)))
        big.append(ApEn(rand_big, m=2, r=0.2*np.std(rand_big)))

    print(len(big))
    print("remain data",ApEn(remain_data, m=2, r=0.2*np.std(remain_data)))
    print("large random",np.average(big))
    print("small random",np.average(small))
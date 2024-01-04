import numpy as np

k1 = 1
prev_average = 0
def avgFilter(x):
    global k1, prev_average
    alpha = (k1-1) / k1
    average = alpha * prev_average + (1-alpha) * x
    prev_average = average
    k1 += 1
    return average

prevX1 = 0
firstRun1 = True
def LPF1(x, alpha):
    global prevX1, firstRun1
    if firstRun1:
        prevX1 = x
        firstRun1 = False
    #alpha = 0.7
    xlpf = alpha * prevX1 + (1 - alpha) * x
    prevX1 = xlpf
    return xlpf



n = 0
xbuf = []
firstRun = True
def movAvgFilter(x):  # for batch
    global n, xbuf, firstRun
    if firstRun:
        n = 10
        xbuf = x * np.ones(n)
        firstRun = False
    else:
        for i in range(n-1):
            xbuf[i] = xbuf[i+1]
        xbuf[n-1] = x
    avg = np.sum(xbuf) / n
    return avg
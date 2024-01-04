import numpy as np
import matplotlib.pyplot as plt
# pip install plotnine
import pandas as pd
from plotnine import aes, facet_grid, geom_point, ggplot
from plotnine import *
from uFunc import avgFilter, movAvgFilter

# initialize
# firstRun = False
# if firstRun == True:
#     k = 1
#     prev_average = 0
# else:
#     firstRun = True

k = 1
prev_average = 0
# recursive expression for averaging filter
def avgFilter(x):
    global k
    global prev_average
    alpha = (k-1) / k
    average = alpha * prev_average + (1-alpha) * x
    prev_average = average
    k += 1
    return average

# -----EX 1-1-----
for i, j in enumerate([10,20,30], start=1):

    print(f'{i}th average = {avgFilter(j)}')

# 1th average = 10.0
# 2th average = 15.0
# 3th average = 20.0





# -----EX 1-2-----
# make noise
def getVolt():
    return 14.4 + np.random.normal(0, 4, 1)


time = np.arange(0, 10, 0.2)

Nsamples = len(time)
Xmsaved  = np.zeros(Nsamples)
Avgsaved = np.zeros(Nsamples)
moving_Avgsaved = np.zeros(Nsamples)

for i in range(Nsamples):
    xm  = getVolt()
    avg = avgFilter(xm)
    mv = movAvgFilter(xm)

    Xmsaved[i]  = xm
    Avgsaved[i] = avg
    moving_Avgsaved[i] = mv



plt.plot(time, Xmsaved, 'b*--', label='Measured')
plt.plot(time, Avgsaved, 'ro',  label='Average')
plt.legend(loc='upper left')
plt.ylabel('Volt [V]')
plt.xlabel('Time [sec]')
# plt.savefig('result/01_avgFilter.png')
plt.show()


ggplot() + \
geom_point(aes(x='time',y='Xmsaved'), color='blue')+ \
geom_path(aes(x='time',y='Xmsaved'),   color='blue', linetype='dashed') + \
geom_point(aes(x='time',y='Avgsaved'), color="red")  + \
geom_path(aes(x='time',y='Avgsaved'),  color="red")  + \
geom_point(aes(x='time',y='moving_Avgsaved'), color="pink") + \
geom_path(aes(x='time',y='moving_Avgsaved'),  color="pink")  + \
xlab('Time [sec]') + ylab('Volt [V]') + ggtitle('Measured vs Average') + \
theme_bw() + \
theme(plot_title=element_text(family='D2Coding', size=20, face='bold', color='black'))



dd = pd.DataFrame({'time':time, 'Xmsaved':Xmsaved, 'Avgsaved':Avgsaved})
ggplot(data=dd) + \
geom_point(aes(x='time',y='Xmsaved'),  color='blue') + \
geom_path(aes(x='time',y='Xmsaved'),   color='blue') + \
geom_point(aes(x='time',y='Avgsaved'), color="red") 
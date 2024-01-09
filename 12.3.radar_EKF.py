import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

A, Q, R = None, None, None
x, P = None, None
firstRun = True

def Hjacob(xp):
    H = np.zeros([1,3])
    x1 = xp[0]
    x3 = xp[2]
    H[:,0] = x1 / np.sqrt(x1**2 + x3**2)
    H[:,1] = 0
    H[:,2] = x3 / np.sqrt(x1**2 + x3**2)
    return H

def hx(xhat):
    x1 = xhat[0]
    x3 = xhat[2]

    zp = np.sqrt(x1**2 + x3**2)
    return zp

'''
시스템 모델 :  (선형)
x = [pos, vel, alt] = [수평거리, 이동속도, 고도]
  = [[0, 1, 0], [0, 0, 0], [0, 0, 0]] @ [pos, vel, alt].T + [0, w1, w2].T

측정 모델 :  (비선형)
z = radar distance
  = np.sqrt(pos**2 + alt**2) + v
'''
def RadarEKF(z, dt):
    global firstRun
    global A, Q, R, x, P
    if firstRun:
        A = np.eye(3) + dt * np.array([[0,1,0],[0,0,0],[0,0,0]])
        Q = np.array([[0,0,0],[0,0.001,0],[0,0,0.001]])
        R = 10
        x = np.array([0,90,1100]).transpose()
        P = 10 * np.eye(3)
        firstRun = False
    else:
        H = Hjacob(x)
        X_pred = A @ x                          # X_pred : State Variable Prediction
        P_pred = A @ P @ A.T + Q                # Error Covariance Prediction

        K = (P_pred @ H.T) @ inv(H@P_pred@H.T + R) # K : Kalman Gain

        x = X_pred + K@(np.array([z - hx(X_pred)])) # Update State Variable Estimation
        P = P_pred - K@H@P_pred # Update Error Covariance Estimation

    pos = x[0]
    vel = x[1]
    alt = x[2]
    return pos, vel, alt


# ------ Test program

dt = 0.05
t = np.arange(0, 20, dt)
Nsamples = len(t)
Xsaved = np.zeros([Nsamples,3])
Zsaved = np.zeros([Nsamples,1])
Estimated = np.zeros([Nsamples,1])


pos_pred = None
def GetRadar(dt):
    global pos_pred
    if pos_pred == None:
        pos_pred = 0

    vel = 100 + 5*np.random.randn()
    pos = pos_pred + vel*dt

    alt = 1000 + 10*np.random.randn()
    v = 0 + pos*0.05*np.random.randn()
    r = np.sqrt(pos**2 + alt**2) + v

    pos_pred = pos
    return r


for k in range(Nsamples):
    r = GetRadar(dt)
    print(r)

    pos, vel, alt = RadarEKF(r, dt)

    Xsaved[k] = [pos, vel, alt]
    Zsaved[k] = r
    Estimated[k] = hx([pos, vel, alt])

PosSaved = Xsaved[:,0]
VelSaved = Xsaved[:,1]
AltSaved = Xsaved[:,2]

t = np.arange(0, Nsamples * dt ,dt)

fig = plt.figure()
plt.subplot(313)
plt.plot(t, PosSaved)
plt.xlabel('Time [Sec]')
plt.ylabel('Position [m]')

plt.subplot(311)
plt.plot(t, VelSaved)
plt.xlabel('Time [Sec]')
plt.ylabel('Speed [m/s]')

plt.subplot(312)
plt.plot(t, AltSaved)
plt.xlabel('Time [Sec]')
plt.ylabel('Altitude [m]')
fig.savefig('result/12_RadarEKF.png')
#plt.show()

plt.figure()
plt.plot(t, Zsaved, 'r--', label='Measured')
plt.plot(t, Estimated, 'b-', label='Estimated')
plt.xlabel('Time [Sec]')
plt.ylabel('Radar distance [m]')
plt.legend(loc='upper left')
plt.savefig('result/12_RadarEKF(2).png')
plt.show()
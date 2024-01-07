# 속도(시간당 이동거리의 변화)가 일정하지 않은 경우.
# 위치로 속도 추정 

# 시스템의 기동은 많이 다르지만 시스템 모델은 같다.
# 왜냐면 거리와 속도의 물리적인 관계는 변하지않기때문입니다. (속도 = 거리/시간)

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy import io

'''
알고싶은것 = '속도'

관심있는 물리량: 상태변수 = x = {위치, 속도}
                                       거리(위치)= 속도 * 시간 
시스템모델 : x(k+1) = Ax(k) + w(k) , w(k) ~ N(0, Q)
  측정모델 :   z(k) = Hx(k) + v(k) , v(k) ~ N(0, R)
'''
# ---- 변수
firstRun = True
x, P = np.array([[0,0]]).T, np.zeros((2,2)) # X : Previous State Variable Estimation, P : Error Covariance Estimation
A, H = np.zeros((2,2)),     np.zeros((1,2))
Q, R = np.zeros((2,2)), 0

def DvKalman(z):
    global firstRun
    global A, Q, H, R
    global x, P
    if firstRun:
        dt = 0.02
        A, Q = np.array([[1, dt], [0, 1]]), np.array([[1, 0], [0, 3]])
        H, R = np.array([[0, 1]]),          np.array([10])
        x = np.array([0, 80]).T
        P = 5 * np.eye(2)
        firstRun = False
    else:
        # Prediction
        x_pred = A@x                              # x_pred : State Variable Prediction
        P_pred = A@P@A.T + Q                      # Error Covariance Prediction
        # Update
        K = (P_pred@H.T) @ inv(H@P_pred@H.T + R)  # K : Kalman Gain
        x = x_pred + K@(z - H@x_pred)             # Update State Variable Estimation
        P = P_pred - K@H@P_pred                   # Update Error Covariance Estimation
    return x

# ------ Test program

# np.random.seed(666)
# Posp, Velp = None, None
# def getPosSensor():
#     global Posp, Velp
#     if Posp == None:
#         Posp = 0
#         Velp = 80
#     dt = 0.1

#     w = 0 + 10 * np.random.normal()
#     v = 0 + 10 * np.random.normal()

#     z = Posp + Velp * dt + v  # Position measurement

#     Posp = z - v
#     Velp = 80 + w
#     return z, Posp, Velp

# 초음파 측정기로 측정한 거리
input_mat = io.loadmat('./SonarAlt.mat')
def getSonar(i):
    z = input_mat['sonarAlt'][0][i]  # (1, 1501)
    return z

Nsamples = 500
time = np.arange(0, Nsamples/50, 0.02)

X_esti  = np.zeros([Nsamples, 2])
Z_saved = np.zeros([Nsamples,2])

for i in range(Nsamples):
    Z, pos_true, vel_true = getSonar(i)
    pos, vel = DvKalman(Z)

    X_esti[i] = [pos, vel]
    Z_saved[i] = [pos_true, vel_true]

plt.figure()
plt.plot(time, Z_saved[:,0], 'b.', label = 'Measurements')
plt.plot(time, X_esti[:,0], 'r-', label='Kalman Filter')
plt.legend(loc='upper left')
plt.ylabel('Position [m]')
plt.xlabel('Time [sec]')
# plt.savefig('result/09_DvKalman-Position.png')

plt.figure()
plt.plot(time, Z_saved[:,1], 'b--', label='True Speed')
plt.plot(time, X_esti[:,1], 'r-', label='Kalman Filter')
plt.legend(loc='upper left')
plt.ylabel('Velocity [m/s]')
plt.xlabel('Time [sec]')
#plt.savefig('result/09_DvKalman-Velocity.png')
plt.show()
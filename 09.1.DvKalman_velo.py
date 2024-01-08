# 저주파통과필터외는 차원이 다른칼만필터의 진짜능력
# 신형열차성능검사 - 직선선로에서 80m/s속도 유지확인
# 속도데이터 모름. 위치정보만 있음
# 속도 = 거리/시간

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

'''
측정된 위치로, 속도를 추정

관심있는 물리량: 상태변수 = x = {위치, 속도} = {pos, vel} = {측정, ?} ={1, 0}
시스템모델 : x(k+1) = Ax(k) + w(k) , w(k) ~ N(0, Q)
  측정모델 :   x_meas(k) = Hx(k) + v(k) , v(k) ~ N(0, R)
'''

firstRun = True
x = np.array([[0,0]]).T  # X : Previous State Variable Estimation, {위치, 속도}
P = np.zeros((2,2))      # P : Error Covariance Estimation
A, H = np.array([[0,0], [0,0]]), np.array([[0,0]])
Q, R = np.array([[0,0], [0,0]]), 0

def DvKalman(x_meas):
    global firstRun
    global A, Q, H, R
    global x, P
    if firstRun:
        dt = 0.1
        A = np.array([[1, dt], 
                      [0,  1]])
        Q = np.array([[1,  0], 
                      [0, 3]])
        H = np.array([[1, 0]]),  
        R = np.array([10])

        #x = np.array([0, 20]).transpose()
        x = np.array([0, 80]).T

        P = 5 * np.eye(2)
        firstRun = False
    else:
        x_pred = A@x                              # x_pred : State Variable Prediction
        P_pred = A@P@A.T + Q                      # Error Covariance Prediction
        K = (P_pred@H.T) @ inv(H@P_pred@H.T + R)  # K : Kalman Gain
        x = x_pred + K@(x_meas - H@x_pred)        # Update State Variable Estimation
        P = P_pred - K@H@P_pred                   # Update Error Covariance Estimation
    return x

# ------ Test program

np.random.seed(666)
posi_pred, velo_pred = None, None
def getPosSensor():
    global posi_pred, velo_pred
    if posi_pred == None:
        posi_pred = 0
        velo_pred = 80
    dt = 0.1

    w = 0 + 10 * np.random.normal()
    v = 0 + 10 * np.random.normal()

    x_meas = posi_pred + (velo_pred * dt) + v  # Position measurement

    posi_pred = x_meas - v
    velo_pred = 80 + w
    return x_meas, posi_pred, velo_pred

time = np.arange(0, 10, 0.1)
Nsamples = len(time)

X_esti_saved  = np.zeros([Nsamples, 2])
X_meas_saved = np.zeros([Nsamples, 2])

for i in range(Nsamples):
    Z, pos_true, vel_true = getPosSensor()
    pos, vel = DvKalman(Z)

    X_esti_saved[i] = [pos, vel]
    X_meas_saved[i] = [pos_true, vel_true]

plt.figure()
plt.plot(time, X_meas_saved[:,0], 'b.', label = 'Measurements')
plt.plot(time, X_esti_saved[:,0],  'r-', label='Kalman Filter')  # pos
plt.legend(loc='upper left')
plt.ylabel('Position [m]')
plt.xlabel('Time [sec]')
# plt.savefig('result/09_DvKalman-Position.png')

plt.figure()
plt.plot(time, X_meas_saved[:,1], 'b--', label='True Speed')
plt.plot(time, X_esti_saved[:,1], 'r-', label='Kalman Filter')
plt.legend(loc='upper left')
plt.ylabel('Velocity [m/s]')
plt.xlabel('Time [sec]')
#plt.savefig('result/09_DvKalman-Velocity.png')
plt.show()
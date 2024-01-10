import numpy as np
import matplotlib.pyplot as plt

from math import sin, asin, cos, tan, pi
from scipy import io

# 헬기의 Angular velocity 각속도(p, q, r)를 측정하여 오일러각 (phi, theta, psi)를 추정
prevPhi, prevTheta, prevPsi = None,None,None

def eulerGyro(ax, ay, az):
    g = 9.8
    theta = asin(ax / g)
    phi = asin(-ay / (g * cos(theta)))
    return phi, theta

# ------ Test program
input_mat = io.loadmat('./11_ars/ArsAccel.mat') #fx, fy, fz 
def getGyro(i):
    """Measure angular velocity using gyro"""
    ax = input_mat['fx'][i][0]  # (41500, 1)
    ay = input_mat['fy'][i][0]  # (41500, 1)
    az = input_mat['fz'][i][0]  # (41500, 1)
    return ax, ay, az

n_samples = 41500
dt = 0.01
time = np.arange(0, n_samples*dt ,dt)

EulerSaved = np.zeros([n_samples,3])
for k in range(n_samples):
    p,q,r = getGyro(k)
    # phi, theta, psi = eulerGyro(p,q,r, dt)
    EulerSaved[k] = [p, q, r]

EulerSaved = np.zeros([n_samples,2])
for k in range(n_samples):
    p,q,r = getGyro(k)
    phi, theta = eulerGyro(p,q,r)
    EulerSaved[k] = [phi, theta]

'''
roll  -> phi
pitch -> theta
yaw   -> psi
'''

PhiSaved   = EulerSaved[:,0] * 180/pi
ThetaSaved = EulerSaved[:,1] * 180/pi
PsiSaved   = EulerSaved[:,2] * 180/pi

plt.figure()
plt.plot(time, PhiSaved)
plt.xlabel('Time [Sec]')
plt.ylabel('Roll ($\phi$) angle [deg]')
# plt.savefig('result/11_EulerGyro_roll.png')

plt.figure()
plt.plot(time, ThetaSaved)
plt.xlabel('Time [Sec]')
plt.ylabel('Pitch ($\\theta$) angle [deg]')
# plt.savefig('result/11_EulerGyro_pitch.png')

plt.figure()
plt.plot(time, PsiSaved)
plt.xlabel('Time [Sec]')
plt.ylabel('Yaw ($\\psi$) angle [deg]')
# plt.savefig('result/11_EulerGyro_pitch.png')
plt.show()
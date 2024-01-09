import numpy as np
import matplotlib.pyplot as plt

from math import sin, cos, tan, pi
from scipy import io

# 헬기의 Angular velocity 각속도(p, q, r)를 측정하여 오일러각 (phi, theta, psi)를 추정
prevPhi, prevTheta, prevPsi = None,None,None

def EulerGyro(p,q,r,dt):
    """Calculate Euler angle (Pose Orientation)."""
    global prevPhi, prevTheta, prevPsi
    if prevPhi is None:
        prevPhi   = 0
        prevTheta = 0
        prevPsi   = 0

    sinPhi   = sin(prevPhi)
    cosPhi   = cos(prevPhi)
    cosTheta = cos(prevTheta)
    tanTheta = tan(prevTheta)

    phi   = prevPhi   + dt*(p + q*sinPhi*tanTheta + r*cosPhi*tanTheta)
    theta = prevTheta + dt*(    q*cosPhi          - r*sinPhi)
    psi   = prevPsi   + dt*(    q*sinPhi/cosTheta + r*cosPhi/cosTheta)

    prevPhi   = phi
    prevTheta = theta
    prevPsi   = psi

    return phi, theta, psi


# ------ Test program
input_mat = io.loadmat('./11_ars/ArsGyro.mat') # wx,wy,wz
def getGyro(i):
    """Measure angular velocity using gyro"""
    p = input_mat['wx'][i][0]  # (41500, 1)
    q = input_mat['wy'][i][0]  # (41500, 1)
    r = input_mat['wz'][i][0]  # (41500, 1)
    return p, q, r


n_samples = 41500
dt = 0.01
time = np.arange(0, n_samples*dt ,dt)

EulerSaved = np.zeros([n_samples,3])
for k in range(n_samples):
    p,q,r = getGyro(k)
    # phi, theta, psi = EulerGyro(p,q,r, dt)
    EulerSaved[k] = [p, q, r]

EulerSaved = np.zeros([n_samples,3])
for k in range(n_samples):
    p,q,r = getGyro(k)
    phi, theta, psi = EulerGyro(p,q,r, dt)
    EulerSaved[k] = [phi, theta, psi]

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
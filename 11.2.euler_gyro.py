import numpy as np
import matplotlib.pyplot as plt

from math import sin, cos, tan, pi
from scipy import io

'''
오일러 각도의 변화율을 나타내는 식
[phi,theta,psi] = [[1,  sin(phi)tan(theta), cos(phi)tan(theta)],
                   [0,            cos(phi), -sin(theta)],
                   [0, sin(phi)/cos(theta), cos(phi)/con(theta)]]@[p,q,r].T
속도를 적분하면 위치를, 가속도를 적분하면 속도를 구할 수 있습니다. 
이와 마찬가지로, 각속도를 적분하면 각도를 구할 수 있습니다.

이를 적분하여 오일러 각도를 구하려면, 
각 변화율에 대해 시간 간격(dt)을 곱한 후 
이전 각도에 더하는 방식으로 수행할 수 있습니다.

이 함수는 각속도(p, q, r)와 시간 간격(dt)을 입력으로 받아, 
오일러 각도(phi, theta, psi)를 계산합니다. 
계산된 각도는 전역 변수(prevPhi, prevTheta, prevPsi)에 저장되어, 다음 호출 시에 사용됩니다. 
아래 함수 EulerGyro는 계산된 오일러 각도를 반환합니다.
'''
# 헬기의 Angular velocity 각속도(p, q, r)를 측정하여 오일러각 (phi, theta, psi)를 추정
prevPhi, prevTheta, prevPsi = None,None,None

def EulerGyro(p,q,r,dt):
    """Calculate Euler angle (Pose Orientation)."""
    global prevPhi, prevTheta, prevPsi
    if prevPhi is None:
        # 초기 오일러 각도를 0으로 설정 
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

    prevPhi, prevTheta, prevPsi = phi, theta, psi

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

## getGyro 관찰값 
EulerSaved = np.zeros([n_samples, 3])  # [phi, theta, psi]
for k in range(n_samples):
    p,q,r = getGyro(k)
    # phi, theta, psi = EulerGyro(p,q,r, dt)
    EulerSaved[k] = [p, q, r]

EulerSaved = np.zeros([n_samples,3])
for k in range(n_samples):
    p,q,r = getGyro(k)
    phi, theta, psi = EulerGyro(p,q,r, dt)
    EulerSaved[k] = [phi, theta, psi]

# 오일러 각도를 라디안에서 도(degree)로 변환
PhiSaved   = EulerSaved[:,0] * 180/pi
ThetaSaved = EulerSaved[:,1] * 180/pi
PsiSaved   = EulerSaved[:,2] * 180/pi

'''
roll  -> phi
pitch -> theta
yaw   -> psi
'''
plt.figure()
plt.plot(time/60, PhiSaved)
plt.xlabel('Time [minites]')
plt.ylabel('Roll ($\phi$) angle [deg]')
# plt.savefig('result/11_EulerGyro_roll.png')

plt.figure()
plt.plot(time/60, ThetaSaved)
plt.xlabel('Time [minites]')
plt.ylabel('Pitch ($\\theta$) angle [deg]')
# plt.savefig('result/11_EulerGyro_pitch.png')

plt.figure()
plt.plot(time, PsiSaved)
plt.xlabel('Time [Sec]')
plt.ylabel('Yaw ($\\psi$) angle [deg]')
# plt.savefig('result/11_EulerGyro_pitch.png')
plt.show()

#자이로는 자세각 "측정"보다는 자세각의 "동태"를 측정하는 데 더 유용
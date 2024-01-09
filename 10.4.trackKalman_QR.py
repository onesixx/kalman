import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# pip install opencv-python scikit-image
import cv2
from skimage.metrics import structural_similarity

np.random.seed(0)
firstRun = True
X, P, A, H, Q, R = 0, 0, 0, 0, 0, 0

def TrackKalman(xm, ym):
    global firstRun
    global A, Q, H, R
    global X, P
    if firstRun:
        dt = 1
        A = np.array([[1,dt, 0, 0], 
                      [0, 1, 0, 0], 
                      [0, 0, 1,dt],
                      [0, 0, 0, 1]])
        Q = np.eye(4)

        H = np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0]])
        R = np.array([[50, 0],
                      [ 0,50]])

        X = np.array([0,0,0,0]).transpose()
        P = 100 * np.eye(4)
        firstRun = False
    '''
    시스템모델 (esti): x(k+1)= Ax(k) + w(k) , w(k) ~ N(0, Q)
      측정모델 (meas): z(k)  = Hx(k) + v(k) , v(k) ~ N(0, R)
    '''
    X_pred = A @ X              # X_pred : State Variable Prediction
    P_pred = A @ P @ A.T + Q    # Error Covariance Prediction

    K = (P_pred @ H.T) @ inv(H@P_pred@H.T + R) # K : Kalman Gain

    z = np.array([xm, ym]).transpose()
    X = X_pred + K@(z - H@X_pred)         # Update State Variable Estimation
    P = P_pred - K@H@P_pred               # Update Error Covariance Estimation

    xh = X[0]
    yh = X[2]

    return xh, yh

# plt.imshow(imageB)
# plt.imshow(grayB)


def GetBallPos(iimg=0):
    """Return measured position of ball by comparing with background image file.
        - References:
        (1) Data Science School:
            https://datascienceschool.net/view-notebook/f9f8983941254a34bf0fee42c66c5539
        (2) Image Diff Calculation:
            https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python
    """
    # Read images.
    imageA = cv2.imread('./10_Img/bg.jpg')
    imageB = cv2.imread(f'./10_Img/{iimg+1}.jpg')

    # Convert the images to grayscale.
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # Compute the Structural Similarity Index (SSIM) between the two images,
    # ensuring that the difference image is returned.
    _, diff = structural_similarity(grayA, grayB, full=True)
    diff = (diff * 255).astype('uint8')

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    M = cv2.moments(contours[0])
    xc = int(M['m10'] / M['m00'])  # center of x as true position.
    yc = int(M['m01'] / M['m00'])  # center of y as true position.

    v = np.random.normal(0, 15)  # v: measurement noise of position.

    xpos_meas = xc + v  # x_pos_meas: measured position in x (observable).
    ypos_meas = yc + v  # y_pos_meas: measured position in y (observable).
    print(f'xpos_meas={xpos_meas}, ypos_meas={ypos_meas}')
    
    return np.array([xpos_meas, ypos_meas])


NoOfImg = 24
Xmsaved = np.zeros([NoOfImg,2])
Xhsaved = np.zeros([NoOfImg,2])

for k in range(NoOfImg):
    xm, ym = GetBallPos(k)
    xh, yh = TrackKalman(xm, ym)

    Xmsaved[k] = [xm, ym]
    Xhsaved[k] = [xh, yh]

plt.figure()
plt.plot(Xmsaved[:,0], Xmsaved[:,1], '*', label='Measured')
plt.plot(Xhsaved[:,0], Xhsaved[:,1], 's', label='Kalman Filter')
plt.legend(loc='upper left')
plt.xlabel('Horizontal [pixel]')
plt.xlim([0, 350])
plt.ylabel('Vertical [pixel]')
plt.ylim([0, 250])
plt.gca().invert_yaxis()
# plt.savefig('result/10_TrackerKalman.png')
plt.show()
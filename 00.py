import numpy as np
import matplotlib.pyplot as plt

# Kalman filter parameters
A = 1.0  # System matrix
H = 1.0  # Measurement matrix
Q = 0.01  # Process noise covariance
R = 1.0  # Measurement noise covariance

x0 = 0.0  # Initial state estimate
P0 = 1.0  # Initial covariance estimate

# Generate true signal and noisy measurements
np.random.seed(0)
n_samples = 100
true_signal = np.sin(np.linspace(0, 2 * np.pi, n_samples))
measurements = true_signal + np.random.normal(0, np.sqrt(R), n_samples)

# Initialize Kalman filter
x = x0
P = P0
filtered_states = []

# Kalman filter loop
for z in measurements:
    # Predict
    x_pred = A * x
    P_pred = A * P * A + Q

    # Update
    y = z - H * x_pred
    S = H * P_pred * H + R
    K = P_pred * H / S
    x = x_pred + K * y
    P = (1 - K * H) * P_pred

    filtered_states.append(x)

# Plot true signal, measurements, and filtered states
plt.figure()
plt.plot(range(n_samples), true_signal, label='True Signal')
plt.scatter(range(n_samples), measurements, marker='x', color='r', label='Measurements')
plt.plot(range(n_samples), filtered_states, label='Filtered States')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

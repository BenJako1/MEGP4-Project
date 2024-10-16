import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
L = 0.4       # Length of the hand-forearm model (m)
m = 2.0       # Mass of the hand-forearm model (kg)
g = 9.81      # Gravitational acceleration (m/s^2)
I = (1/3) * m * L**2  # Moment of inertia of the hand-forearm
M0 = 0.0      # Constant moment exerted by the wrist (Nm)

# Equation of motion: d2theta/dt2 = -mgL/2 * sin(theta) / I + M0/I
def equation_of_motion(t, y):
    theta, omega = y  # y = [theta, dtheta/dt]
    dydt = [omega, -(m * g * L / 2) * np.sin(theta) / I + M0 / I]
    return dydt

# Initial conditions
theta0 = np.pi / 4  # Initial angle (radians)
omega0 = 0.0        # Initial angular velocity (rad/s)
y0 = [theta0, omega0]

# Time points where the solution is computed
t_span = (0, 10)    # Simulation time from 0 to 10 seconds
t_eval = np.linspace(t_span[0], t_span[1], 1000)  # 1000 points in time

# Solve the differential equation
sol = solve_ivp(equation_of_motion, t_span, y0, t_eval=t_eval)

# Extract the results
theta = sol.y[0]    # Theta over time
omega = sol.y[1]    # Angular velocity over time
t = sol.t           # Time

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(t, theta, label="Theta (rad)")
plt.plot(t, omega, label="Angular velocity (rad/s)")
plt.title("Motion of the Hand Model with Wrist Moment")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
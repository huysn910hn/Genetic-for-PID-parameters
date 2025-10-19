import control
import numpy as np
K = 2
T = 1
a = 12
b = 20.02
num = [K]
den = [T, a, b]
plant = control.TransferFunction(num, den)
print(plant)
def simulate_step_response(Kp, Ki, Kd, t_end=10, num_points=100):
    pid = control.TransferFunction([Kd, Kp, Ki], [1, 0])
    sys = control.series(pid, plant)
    sys = control.feedback(sys, 1)
    t = np.linspace(0, t_end, num_points)
    t, y = control.step_response(sys, t)
    error = 1 - y
    return t, y, error

def f_IAE(error, t):
    IAE = np.trapz(np.abs(error), t)
    return IAE

def f_ITAE(error, t):
    ITAE = np.trapz(t * np.abs(error), t)
    return ITAE

def f_MSE(error, t):
    MSE = np.mean(error**2)
    return MSE


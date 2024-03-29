from matplotlib.widgets import Slider
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import random

fig = plt.figure()
ax1 = fig.add_subplot(211, xlim=(0, 50), ylim=(0, 200))
ax2 = fig.add_subplot(212, xlim=(0, 50), ylim=(-90, 100))

# Add a slider to select the input value
slider_ax = fig.add_axes([0.2, 0.01, 0.6, 0.05])
slider = Slider(slider_ax, 'Input', 0, 150, valinit=100)

max_point = 50
line, = ax1.plot(np.arange(max_point), np.ones(
    max_point, dtype=float)*np.nan, lw=2)
line2, = ax2.plot(np.arange(max_point), np.ones(
    max_point, dtype=float)*(-70), lw=2)
line3, = ax2.plot(np.arange(max_point), np.ones(
    max_point, dtype=float)*(-20), lw=2)

paused = False  # Initialize pause flag to False


def get_data_func():
    noise = np.random.normal(0, 10) # Add random noise to the input value
    return slider.val + noise


def izhikevich(a, b, c, d):
    def model(old_Vu, I):
        V, u = old_Vu
        dV_dt = 0.04 * V ** 2 + 5 * V + 140 - u + I
        du_dt = a * (b * V - u)
        if V + dV_dt >= 30:  # Check if neuron has spiked
            V = c
            u = u + d
        else:
            V = V + dV_dt
            u = u + du_dt
        return [V, u]
    return model


model = izhikevich(0.02, 0.2, -65, 2)


def animate(i):
    global paused
    y = 0
    if paused:
        y = get_data_func()
    old_y = line.get_ydata()
    new_y = np.r_[old_y[1:], y]
    line.set_ydata(new_y)

    old_V = line2.get_ydata()
    old_u = line3.get_ydata()
    V, u = model([old_V[-1], old_u[-1]], y)
    new_V = np.r_[old_V[1:], V]
    new_u = np.r_[old_u[1:], u]
    line2.set_ydata(new_V)
    line3.set_ydata(new_u)
    # Add a red vertical line whenever the threshold is passed

    return line, line2, line3


def on_space_press(event):
    global paused
    if event.key == ' ':
        paused = True


def on_space_release(event):
    global paused
    if event.key == ' ':
        paused = False


anim = animation.FuncAnimation(
    fig, animate,
    interval=10
)

fig.canvas.mpl_connect('key_press_event', on_space_press)
fig.canvas.mpl_connect('key_release_event', on_space_release)

plt.show()

import numpy as np
import matplotlib.pyplot as plt

c = 1.0
length = 1.0
spatial_points = 101
spatial_step = length / (spatial_points - 1)
time_step = 0.001
total_time = 10.0


x = np.linspace(0, length, spatial_points)

time_steps = int(total_time / time_step)

Psi = np.exp(-((x - length/2)**2) / 0.1)

Psi_grid = np.zeros((time_steps, spatial_points))

Psi_prev = Psi.copy()  
Psi_next = np.zeros(spatial_points) 


plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(x, Psi)
ax.set_ylim(-1.2, 1.2)
ax.set_xlim(0, length)



for n in range(time_steps):
    for i in range(1,spatial_points-1):
        Psi_next[i] = c**2 * time_step**2 / spatial_step**2 * (Psi[i+1] - 2*Psi[i] + Psi[i-1]) + 2*Psi[i] - Psi_prev[i]
    
    Psi_prev = Psi.copy()
    Psi = Psi_next.copy()

    Psi_grid[n, :] = Psi

    if n % 10 == 0:
        line.set_ydata(Psi)
        plt.pause(0.01)


plt.ioff()
plt.show()


plt.figure()
plt.imshow(Psi_grid, aspect='auto', cmap='hot', extent=[0, length, total_time, 0])
plt.colorbar(label='Psi')
plt.show()
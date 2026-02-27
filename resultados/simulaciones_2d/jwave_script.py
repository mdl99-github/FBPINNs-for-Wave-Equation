import time

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from scipy import ndimage

from jax import jit
from jax import numpy as jnp
from matplotlib import pyplot as plt

from jwave.geometry import TimeAxis
from jwave.geometry import Medium
from jwave.geometry import Domain
from jwave.geometry import Sensors, circ_mask, points_on_circle


from jwave import FourierSeries
from jwave.acoustics import simulate_wave_propagation
from jwave.geometry import *
from jwave.geometry import circ_mask
from jwave.utils import show_field

im = Image.open('tejido_pequeno0.png') 

im = -(np.array(im.convert('L'))[:,8:-14]/255 - 1)

dx = 15e-3/im.shape[0]

plt.imshow(im)

N = int(0.1/dx)

start = time.time()

#-----------------------jwave--------------------------------

domain = Domain((N, N), (0.1/N, 0.1/N))
medium = Medium(domain=domain, sound_speed=1490.0, attenuation=0.0, pml_size=30.0)

time_axis = TimeAxis.from_medium(medium, cfl=0.3, t_end=100e-6)
print(f"El dominio espacial en x va de {domain.spatial_axis[0].min()} m a {domain.spatial_axis[1].max()} m")
print(f"El dominio espacial en y va de {domain.spatial_axis[1].min()} m a {domain.spatial_axis[1].max()} m")

p0 = np.pad(im, (int((N-im.shape[0])/2), int((N-im.shape[0])/2)))
print(p0.shape)

p0 = FourierSeries(p0, domain)

show_field(p0)
plt.title("Initial pressure")
plt.show()

num_sensors = 64*2
rx = 45e-3
ri = rx/domain.dx[0]
xi, yi = points_on_circle(num_sensors, ri, (int(domain.N[0]/2), int(domain.N[0]/2)))
sensors_positions = (xi, yi)
sensors = Sensors(positions=sensors_positions)

@jit
def compiled_simulator(medium, p0):
    a = simulate_wave_propagation(medium, time_axis, p0=p0, sensors=sensors)
    return a

p = compiled_simulator(medium, p0)
sensors_data = compiled_simulator(medium, p0)[..., 0]

end = time.time()

print(f"Tardó {end-start} segundos")

np.save('sinogram0.npy', sensors_data)

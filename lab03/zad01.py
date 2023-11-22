import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt
import math
import numpy as np

def endurance(x, y, z, u, v, w):
    return math.exp(-2 * (y - math.sin(x)) ** 2) + \
           math.sin(z * u) + math.cos(v * w)

def endurance_opt(swarm):
    return [-endurance(*s) for s in swarm]

options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

x_max = [2, 2]
x_min = [1, 1]
my_bounds = (x_min, x_max)

print("sphere)")
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=my_bounds)
cost, pos = optimizer.optimize(fx.sphere, iters=1000)

# c)
print("\nendurance")
my_dimensions = 6
my_bounds = (np.zeros(my_dimensions), np.ones(my_dimensions))
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=my_dimensions, options=options, bounds=my_bounds)
cost, pos = optimizer.optimize(endurance_opt, iters=5000)
cost_history = optimizer.cost_history
plot_cost_history(cost_history)
plt.show()

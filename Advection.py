"""
Numerical solving of advection equation with FTCS method (shown to be unstable)
and the Lax-Friedrichs method (shown to be stable, if the Courant condition is respected)
@author: Yilin Wang 


"""

import numpy as np 
from matplotlib import pyplot as plt 

u = -0.1
# defining grid size 
grid_spacing = 0.05
N = 400
gridline = np.arange(0, 400, grid_spacing)

# defining time step and total steps to evolve system: 
timestep = 0.05
total_time = np.arange(0, 100, timestep)

# initializing function, first order derivative, and second order derivative: 
func_0 = gridline.copy()
funcp_0 = np.zeros(len(gridline)-2)
funcpp_0 = np.zeros(len(gridline)-2)
funcp_0 = (func_0[2:] - func_0[:-2]) / (2*grid_spacing) 
funcpp_0= (func_0[2:] - 2 * func_0[1:-1] + func_0[:-2]) / (grid_spacing**2)

# initial states, prepared to be used in the FTCS method: 
func_0_FTCS = func_0.copy()
funcp_0_FTCS = funcp_0.copy()
funcpp_0_FTCS = funcpp_0.copy()

# initial states, prepared to be used in the Lax-Friedrichs method: 
func_0_LF = func_0.copy()
funcp_0_LF = funcp_0.copy()
funcpp_0_LF = funcpp_0.copy()


def take_funcp(func):
	return (func[2:] - func[:-2]) / (2*grid_spacing) 

def take_funcpp(func):
	return (func[2:] - 2 * func[1:-1] + func[:-2]) / (grid_spacing**2)

def func_evolve_FTCS(func, timestep, funcp): 
	func[1:-1] -= u * timestep *funcp
	funcp = take_funcp(func)
	return func, funcp

def func_evolve_LF(func, timestep, funcp, funcpp): 
	func[1:-1] = 0.5*(func[2:]+func[:-2]) - u * 0.5*timestep/grid_spacing * (func[2:]-func[:-2])
	return func, funcp, funcpp 

plt.ion()
fig, axes = plt.subplots(1,2)
axes[0].set_title("FTCS method")
axes[1].set_title("Lax-Friedrichs method")
# initializing plot; 
axes[0].plot(gridline, func_0, 'k-', label="initial cond.")
x1, = axes[0].plot(gridline, func_0_FTCS, "g.", label="FCTS evolved")
axes[1].plot(gridline, func_0, 'k-', label="initial cond.")
x2, = axes[1].plot(gridline, func_0_LF, "g.", label="LF evolved")

axes[0].legend()
axes[1].legend()

fig.canvas.draw()

# FTCS evolvement of function: 
count = 0 
while count < len(total_time): 
	for i in range(1000):
		func_0_FTCS, funcp_0_FTCS = func_evolve_FTCS(func_0_FTCS, timestep, funcp_0_FTCS)
		func_0_LF, funcp_0_LF, funcpp_0_LF = func_evolve_LF(func_0_LF, timestep, funcp_0_LF, funcpp_0_LF)
	x1.set_ydata(func_0_FTCS)
	x2.set_ydata(func_0_LF)
	fig.canvas.draw()
	plt.pause(0.001)
	count = count + 1




"""
Numerical solving of advection diffusion equation with Lax-Friedrichs method, 
respecting the Courant condition, using two different duffision coefficients. 

@author: Yilin Wang 
28 Feb. 2020 
"""

import numpy as np 
import matplotlib.pyplot as plt 


u = -0.1
# defining grid size 
grid_spacing = 1
N = 100
gridline = np.arange(0, N, grid_spacing)

# defining time step and total steps to evolve system: 
timestep = 0.5
total_time = np.arange(0, N, timestep)

# initializing function, first order derivative, and second order derivative: 
func_0 = gridline.copy()
funcp_0 = np.zeros(len(gridline)-2)
funcpp_0 = np.zeros(len(gridline)-2)
funcp_0 = (func_0[2:] - func_0[:-2]) / (2*grid_spacing) 
funcpp_0= (func_0[2:] - 2 * func_0[1:-1] + func_0[:-2]) / (grid_spacing**2)

# diffusion terms
D = [0.1, 10]

# initial states, prepared to be used in the Lax-Friedrichs method, with diffusion
func_0_LFd_1 = func_0.copy()
funcp_0_LFd_1 = funcp_0.copy()
funcpp_0_LFd_1 = funcpp_0.copy()

func_0_LFd_2 = func_0.copy()
funcp_0_LFd_2 = funcp_0.copy()
funcpp_0_LFd_2 = funcpp_0.copy()


def A_func(n, D):
	"""
	Define function to calculate matrix A for updating grid with diffusion term (implicid method)
	Argument(s):
	n - integer 
	D: float
	"""
	beta = D * timestep / (grid_spacing**2)
	A = np.eye(n) * (1.0+2.0*beta) + np.eye(n, k=1) * -beta + np.eye(n, k=-1)*-beta
	return A

def func_evolve_LFd(func, funcp, funcpp, D): 
	"""
	Define fucntion to evolve a function to timestep n+1 using the previous timestep n,
	using the Lax-Friedrichs method with diffusion term
	Arguments: 
	func - array 
	funcp - array
	funcpp - array 
	D - float 
	"""
	func[1:-1] += -u*timestep*funcp+ D*timestep*funcpp
	funcp = (func[2:] - func[:-2]) / (2*grid_spacing)  
	funcpp= (func[2:] - 2 * func[1:-1] + func[:-2]) / (grid_spacing**2)
	return func, funcp, funcpp 


"""plotting the numerical solution's evolvement with each time step, 
using LF methods with diffusiion term, for two different diffusion coefficients  
"""
plt.ion()
fig, axes = plt.subplots(1,2)
axes[0].set_title(f"D={D[0]}")
axes[1].set_title(f"D={D[1]}")
# initializing plot: 
axes[0].plot(gridline, func_0, 'k-', label="initial cord.")
x1, = axes[0].plot(gridline, func_0_LFd_1, "g.", label="LF evolved")
axes[1].plot(gridline, func_0, 'k-', label="initial cond.")
x2, = axes[1].plot(gridline, func_0_LFd_2, "g.", label="LF evolved")

axes[0].legend()
axes[1].legend()

fig.canvas.draw()

count = 0 

# LF evolvement of function for two diffusive coefficients 

while count < len(total_time): 
	for i in range(300):
		A_1 = A_func(len(gridline), D[0])
		A_1[0][0] = 1
		A_1[0][1] = 0 
		A_1[-1][-1] = 1
		A_1[-1][-2] = 0 
		func_0_LFd_1 = np.linalg.solve(A_1, func_0_LFd_1)
		func_0_LFd_1, funcp_0_LFd_1, funcpp_0_LFd_1 = func_evolve_LFd(func_0_LFd_1, funcp_0_LFd_1, funcpp_0_LFd_1, D[0])

		A_2 = A_func(len(gridline), D[1])
		A_2[0][0] = 1
		A_2[0][1] = 0 
		A_2[-1][-1] = 1
		A_2[-1][-2] = 0 
		func_0_LFd_2 = np.linalg.solve(A_2, func_0_LFd_2)
		func_0_LFd_2 ,funcp_0_LFd_2, funcpp_0_LFd_2 = func_evolve_LFd(func_0_LFd_2, funcp_0_LFd_2, funcpp_0_LFd_2, D[0])

		x1.set_ydata(func_0_LFd_1)
		x2.set_ydata(func_0_LFd_2)
		fig.canvas.draw()
		plt.pause(0.001)
		count = count + 1


import numpy as np 
# from Advection import take_funcp, take_funcpp, func_evolve_FTCS, func_evolve_LF
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
	beta = D * timestep / (grid_spacing**2)
	A = np.eye(n) * (1.0+2.0*beta) + np.eye(n, k=1) * -beta + np.eye(n, k=-1)*-beta
	# A[0][0] = 1
	# A[0][1] = 0 
	# A[-1][-1] = 1
	# A[-1][-2] = 0 
	return A

def func_evolve_LFd(func, funcp, funcpp, D): 
	func[1:-1] += -u*timestep*funcp+ D*timestep*funcpp
	funcp = (func[2:] - func[:-2]) / (2*grid_spacing)  
	funcpp= (func[2:] - 2 * func[1:-1] + func[:-2]) / (grid_spacing**2)
	return func, funcp, funcpp 

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








# PHYS432_ProblemSet4

Name: Yilin Wang (260720347)
Python version 3. 
Acknowledgement: Antoine Belley has helped me with understanding coding derivatives; An Vuong has helped me understand interactive plotting with matplotlib.pyplot.ion. Professor Eve Lee helped with the debugging process of Advection.py by pointing out that I was not initializing the plot. I also took the code commenting style from Antoine Belley. 


* Advection.py numerically evolves to solve for the solution to the advection equation using two methods - FTCS and the Lax-Friedrichs (LF) methods. 
** The solution, evolved through iterations of time steps, is shown through the interactive plotting, matplotlib.pyplot.ion, for both methods side by side. It can be seen through the plot's evolvement that the FTCS method is unstable, whereas the LF method is stable (when the Courant condition is respected.)
* Advection_Diffusion.py numerically solves for the advection diffusion equation (similar to the advection equation, but with a diffusion component). 
** Two diffusion coefficients, D = 0.1 and 10, are used and the evolved results for both are compared in the interactive plot side by side with the same timesteps. Both are numerically stable. It can be observed from the plot that the results develop faster for smaller diffusion coefficient. 


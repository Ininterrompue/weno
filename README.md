This repository seeks to make use of the weighted essentially non-oscillatory (WENO) finite-difference schemes (Liu, Osher, and Chan, 1994) (Jiang and Shu, 1997) to solve nonlinear hyperbolic PDEs. Our short-term goals are to implement characteristic decomposition, the AdaWENO (Peng et al. 2018) scheme, and other optimizations to improve runtime while maintaining code readability.

We use the 3rd order TVD Runge-Kutta method for the time discretization and the 5th order WENO scheme for the spatial discretization. The nonlinear weights are calculated using smoothness indicators defined by Yamaleev and Carpenter (2009), which improves upon the accuracy of the scheme at shocks and discontinuous points.

We have written solvers for Burgers' equation and the Euler equations in 1D. Extensions to 2D/3D Cartesian, the Navier-Stokes equations, and the ideal/resistive MHD equations will follow.

## Overview

This repository seeks to make use of the weighted essentially non-oscillatory (WENO) finite-difference schemes<sup>1,2</sup> to solve nonlinear hyperbolic PDEs. Our short-term goals are to implement characteristic decomposition and the AdaWENO scheme<sup>3</sup>, the latter of which aims to reduce the computational expense of the former.

We use the 3rd order TVD Runge-Kutta method for the time discretization and the 5th order WENO scheme for the spatial discretization<sup>4</sup>. The nonlinear weights are calculated using smoothness indicators defined by Yamaleev and Carpenter<sup>5</sup>, which improves upon the accuracy of the scheme at shocks and discontinuous points.

We have written solvers for Burgers' equation and the Euler equations in 1D. Extensions to 2D/3D Cartesian, the Navier-Stokes equations, and the ideal/resistive MHD equations will follow. Because these systems may involve source terms, forming a well-balanced scheme will be of great importance.

## References
1. X. D. Liu, S. Osher, and T. Chan, *J. Comput. Phys.* **115**:200-212 (1994).
2. G. S. Jiang and C. W. Shu, *J. Comput. Phys.* **126**(1):202-228 (1996).
3. J. Peng et al., *Comput Fluids* **179**:34-51 (2019).
4. C. W. Shu, "Essentially non-oscillatory and weighted essentially non-oscillatory schemes for hyperbolic conservation laws," in *Advanced Numerical Approximation of Nonlinear Hyperbolic Equations*, edited by A. Quarteroni, Springer Lecture Notes in Mathematics Vol. 1697 (Springer, New York, 1998).
5. N. K. Yamaleev and M. H. Carpenter, *J. Comput. Phys.* **228**(11):4248-4272 (2009).

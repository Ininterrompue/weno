## Overview

This repository seeks to make use of the weighted essentially non-oscillatory (WENO) finite-difference schemes<sup>1,2</sup> to solve nonlinear hyperbolic PDEs. Our short-term goals are to generalize the AdaWENO scheme<sup>3</sup> to other systems, as well as solve systems with source terms<sup>4</sup>.

We use the 3rd order TVD Runge-Kutta method for the temporal discretization and the 5th order WENO scheme for the spatial discretization<sup>5</sup>. The nonlinear weights are calculated using smoothness indicators defined by Yamaleev and Carpenter<sup>6</sup>, which improves upon the accuracy of the scheme at shocks and discontinuous points as compared with those weights defined in Jiang and Shu<sup>2</sup>.

We have written solvers for Burgers' equation and the Euler equations in 1D and 2D. The following extensions are proposed: shallow water equations<sup>7</sup>, ideal MHD<sup>8,9</sup>, Hall MHD<sup>10</sup>.

## References
1. X. D. Liu, S. Osher, and T. Chan, *J. Comput. Phys.* **115**:200-212 (1994).
2. G. S. Jiang and C. W. Shu, *J. Comput. Phys.* **126**(1):202-228 (1996).
3. J. Peng et al., *Comput. Fluids* **179**:34-51 (2019).
4. Y. Xing and C. W. Shu, *J. Sci. Comput.* **27**(1-3):477-494 (2005).
5. C. W. Shu, "Essentially non-oscillatory and weighted essentially non-oscillatory schemes for hyperbolic conservation laws," in *Advanced Numerical Approximation of Nonlinear Hyperbolic Equations*, Springer Lecture Notes in Mathematics **1697** (Springer, New York, 1998).
6. N. K. Yamaleev and M. H. Carpenter, *J. Comput. Phys.* **228**(11):4248-4272 (2009).
7. Y. Xing and C. W. Shu, *J. Comput. Phys.* **208**:206-227 (2005).
8. A. J. Christlieb, J. A. Rossmanith, and Q. Tang, *J. Comput Phys.* **268**:302-325 (2014).
9. A. J. Christlieb et al., *SIAM J. Sci. Comput.* **40**(4):A2631-A2666 (2017).
10. J. Huba, "Numerical Methods: Ideal and Hall MHD", *Proceedings of ISSS* **7**:26-31 (2005).

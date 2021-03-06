# Overview
This is a solver for the vorticity transport equation. The vorticity equations are the Navier-Stokes equations decomposed into a linear transport equations and a Laplace equation.

<br />
<p align="center">
    
https://user-images.githubusercontent.com/93407190/176039990-26a2a477-12e9-473d-bc98-a27a654b422d.mp4

This result was generated with this solver. It shows the evolution of vorticities.

The vorticity equations are given by : 
$$ \frac{\partial\omega}{\partial t} + (\vec{u}\cdot\nabla)\omega = 0 $$
$$ \nabla \cdot \vec{u} = 0, \nabla \times \vec{u}= \omega $$

with the initial data and boundary conditions:

$$ \omega (0,\vec{x}) = \omega_0(\vec{x}) $$
$$ \vec{u} \rightarrow \vec{0} \quad \text{for} \quad |\vec{x}| \rightarrow \infty $$

The shown results are generated with elliptical initial conditions:
$$ \omega_0(\vec{x}) = \begin{cases} 20 (1 - \frac{|\vec{x}|}{0.8})  & |\vec{x}| < 0.8,\newline 0 & else  \end{cases} $$ 



# Numerical Method
The vorticity equations are solved via a particle method with using kernel-based interpolation. One advantage of particle methods, are their semi-analytical nature. The vorticity can be computed exact and from this, the velocity vector will be computed. 
The evolution of the particles is given by :

$$ \begin{aligned} \frac{d\vec{x}_i}{dt} &= \vec{u}\_{h, \sigma}(t, \vec{x}_i(t)) \newline \frac{d\omega_i}{dt} &= \mu\Delta\omega\_{h,\sigma}(t, \vec{x}_i(t))\newline \end{aligned} $$ 

The viscosity of the Navier-Stokes equations can be modeled using the $\mu$ parameter. The trajectories of the particles can be computed using any Runge-Kutta scheme.

The functions $\vec{u}_{h, \sigma}(t, \vec{x}_i(t)) \ \text{and} \ \omega\_{h,\sigma} (t, \vec{x}_i(t))$ are the interpolator, that are used to evaluate the field where no particles are present.

$$ \begin{aligned} \omega_{h,\sigma}(t,\vec{x}_i(t)) &= \sum\_{i=1}\^{N}\zeta\_\sigma(\vec{x}-\vec{x}_i)c_i \newline \vec{u}\_{h,\sigma}(t, \vec{x}_i(t)) &= \sum\_{i=1}\^{N} \vec{K}\_\sigma(\vec{x}-\vec{x}_i)c_i \end{aligned} $$

The $c_i$ coefficients are the solution to the following eigenvalues problem, which can be interpreted as a sort of least squares problem:

$$ \begin{pmatrix}
        \zeta_??(\vec{x}_1 ??? \vec{x}_1 ) & \zeta_??(\vec{x}_1 ??? \vec{x}_2) & \dots & \zeta_??(\vec{x}_1 ??? \vec{x}_N ) \\
        \zeta_??(\vec{x}_2 ??? \vec{x}_1) & \zeta_??(\vec{x}_2 ??? \vec{x}_2) & \dots & \zeta_??(\vec{x}_2??? \vec{x}_N ) \\
        \vdots  & \vdots  & &\vdots
    \end{pmatrix}
    \begin{pmatrix}
        c_1 \\
        c_2 \\
        \vdots
    \end{pmatrix}
    =
    \begin{pmatrix}
       \omega_1 \\
       \omega_2 \\
       \vdots 
    \end{pmatrix}$$

Where $\zeta_\sigma(\vec{x}) $ is a kernel function. For this I used the Wendland kernel, since it is a nice compromise between complexity and accuracy.
$$ \begin{aligned} \zeta_\sigma(\vec{x}) &:= \zeta (\frac{\vec{x}}{\sigma}) \newline \zeta(\vec{x}) &:= \bar{\zeta}(|\vec{x}|) \newline \bar{\zeta}(|\vec{x}|) &:= (1-r)^6(35r^2+^18r+3) \end{aligned} $$

(TODO add Velocity Kernel)

# Implementation
The implementation uses the parallel nature of the particles to executed parts of the code on CUDA able GPU. This allows for a great increase in speed and precision.

# Results
<br />
<p align="center">
    <img src="./Results/very_high_vis.gif" width="600" height="500">
    <center><em><font size="+1">This flow has a lot of viscosity, therefore the vorticies die off quickly.
  </em></center></font>
</p>
<br />

<br />
<p align="center">
    <img src="./Results/no_vis.gif" width="600" height="500">
    <center><em><font size="+1">This flow has no viscosity and therefore the kinetic energy is conserved.
  </em></center></font>
</p>
<br />

# References
A similar problem was solved in:
>JOURNAL OF COMPUTATIONAL PHYSICS 138, 821???857 (1997) \
>ARTICLE NO . CP975749 \
>Inviscid Axisymmetrization of an Elliptical Vortex" by P. Koumoutsakos

but using a re-meshing particle method.

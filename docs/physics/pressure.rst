=================
Pressure balance
=================

Given that the colony is quasi-static, we can perform a pressure balance across every edge in the colony. This pressure balance can be written as:

.. math::  e_\mathrm{residual} = \underbrace{p_i - p_j - \frac{t}{r}}_\mathrm{(Pressure\, balance\, per\, edge)}
   
where :math:`p_i` and :math:`p_j` are the 2 cells connected to edge :math:`e`. :math:`t` and :math:`r` are the tension and radius of edge :math:`e`. The system of equations for every edge can be formulated as a matrix:

.. math:: G_pp=q
   
where G_p is the matrix of coefficients, p is the matrix of pressures and q is the matrix of edge tensions/edge radii. Since the system of equations is over-determined, this can be solved by adding a normalization constraint (average pressure is 0) as:

.. math:: \begin{bmatrix}G_p^TG_p & C_2^T \\ C_2 & 0\end{bmatrix} \begin{bmatrix}p_1 \\ \vdots \\ p_M \\ \lambda_2 \end{bmatrix} = \begin{bmatrix}q_1 \\ \vdots \\ q_M \\ 0 \end{bmatrix}
   
In DLITE, we formulate this as an optimization problem with an objective function

.. math:: \underset{p}{\mathrm{minimize}} \; g(p) = \sum_{j=1}^E e_{j,\mathrm{residual}}^2

This is solved as an unconstrained optimization problem using initial guesses from the previous time point. 

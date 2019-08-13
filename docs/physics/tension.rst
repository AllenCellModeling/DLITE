=================
Tension balance
=================

The model assumes that any given image of a cell colony is quasi-static. This means that we can perform a force balance at every node in the colony. The force balance per node can be written as:

.. math:: n_\mathrm{residual} = \underbrace{\left| \sum_{i=1}^{e_n} t_i v_i \right|}_\mathrm{(Tension\, balance\, per\, node)}
   
where n is a node, e_n are the edges connected to node n, t is the tension of edge i connected to node n and v is the local tangent vector of edge i connected to node n. This can be formulated as a system of equations:

.. math:: G_\gamma \gamma = 0
   
where G is the matrix of coefficients and gamma is the matrix of edge tensions. Since the system of equations is overdetermined, we can apply a normalization constraint (average tension is 1) to get

.. math:: \begin{bmatrix}G_\gamma^TG_\gamma & C_1^T \\ C_21& 0\end{bmatrix} \begin{bmatrix}\gamma_1 \\ \vdots \\ \gamma_n \\ \lambda_1 \end{bmatrix} = \begin{bmatrix}0 \\ \vdots \\ 0 \\ N \end{bmatrix}

The quality of the tension inference can be determined by tension residuals per node and the condition number of the tension matrix. DLITE uses the same underlying physics, but formulates the system of equations as an unconstrained optimization problem with an objective function defined as

.. math:: \underset{t}{\mathrm{minimize}}\; f(t) = \sum_{j=1}^N \left( n_{j,\mathrm{residual}} + \underbrace{ \frac{n_{\mathrm{j,residual}}} {\sum_{i=1}^{e_{n_j}} \left| t_i v_i \right|}}_\mathrm{Regularizer}\right)

This includes a regularizer to reduce sensitivity to poorly conditioned systems of equations. Further, we use initial guesses of edge tensions from the previous time point. The inferred tensions are then normalized to 1. We use Scipy's Basinhopping global optimization routine to obtain solutions at the first time point and the unconstrained optimizer L-BFGS-B to infer solutions at all subsequent time points.


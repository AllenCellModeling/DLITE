=================
The physics of tension balance
=================

The model assumes that any given image of a cell colony is quasi-static. This means that we can perform a force balance at every node in the colony. The force balance per node can be written as:

.. image:: https://user-images.githubusercontent.com/40371793/61190807-02bfca80-a657-11e9-8113-994576e65efd.jpg
   :scale: 0.01 %
   :align: center
   
where n is a node, e_n are the edges connected to node n, t is the tension of edge i connected to node n and v is the local tangent vector of edge i connected to node n. This can be formulated as a system of equations:

.. image:: https://user-images.githubusercontent.com/40371793/61190873-364f2480-a658-11e9-9c92-ba01a510de27.jpg
   :scale: 30 %
   :align: center
   
Since the system of equations is overdetermined, we can apply a normalization constraint (average tension is 1) to get

.. image:: https://user-images.githubusercontent.com/40371793/61190854-fa1bc400-a657-11e9-8538-0bc889170a68.jpg
   :scale: 30 %
   :align: center

The quality of the tension inference can be determined by tension residuals per node and the condition number of the tension matrix. DLITE uses the same underlying physics, but formulates the system of equations as an unconstrained optimization problem with an objective function defined as

.. image:: https://user-images.githubusercontent.com/40371793/61190904-ce4d0e00-a658-11e9-8f63-ca618a0c0b7d.jpg
   :scale: 30 %
   :align: center

This includes a regularizer to reduce sensitivity to poorly conditioned systems of equations. Further, we use initial guesses of edge tensions from the previous time point. The inferred tensions are then normalized to 1 post-facto. We use Scipy's Basinhopping global optimization routine to obtain solutions at the first time point and the unconstrained optimizer L-BFGS-B to infer solutions at subsequent time points.


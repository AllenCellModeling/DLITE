=================
The physics of pressure balance
=================

Given that the colony is quasi-static, we can perform a pressure balance across every edge in the colony. This pressure balance can be written as:

.. image:: https://user-images.githubusercontent.com/40371793/61191154-94313b80-a65b-11e9-9510-92a1b16af87b.jpg
   :scale: 1 %
   :align: center
   
where p_i and p_j are the 2 cells connected to edge e. t and r are the tension and radius of edge e. The system of equations for every edge can be formulated as a matrix:

.. image:: https://user-images.githubusercontent.com/40371793/61191152-91364b00-a65b-11e9-811d-a867d1c8696f.jpg
   :scale: 1 %
   :align: center
   
where G_p is the matrix of coefficients, p is the matrix of pressures and q is the matrix of edge tensions/edge radii. Since the system of equations is over-determined, this can be solved by adding a normalization constraint (average pressure is 0) as:

.. image:: https://user-images.githubusercontent.com/40371793/61191155-95faff00-a65b-11e9-9d41-a5f54e24dda7.jpg
   :scale: 1 %
   :align: center
   
In DLITE, we formulate this as an optimization problem with an objective function

.. image:: https://user-images.githubusercontent.com/40371793/61191220-731d1a80-a65c-11e9-973e-a7f569f1f68c.jpg
   :scale: 1 %
   :align: center

This is solved as an unconstrained optimization problem using initial guesses from the previous time point. 

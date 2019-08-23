Synthetic Data
==============

The validation of ``DLITE`` on synthetic data requires the production of time series of colonies of cells moving around and generating tension as they do. Producing this data requires several steps, documented below. This is dependent on the installation of `SurfaceEvolver`_. Please see `their documentation`_ for installation instructions.

.. _SurfaceEvolver: http://facstaff.susqu.edu/brakke/evolver/html/evolver.htm
.. _their documentation: http://facstaff.susqu.edu/brakke/evolver/html/install.htm

.. toctree::
   :maxdepth: 2

   Generate_Voronoi_Colony_Mesh
   Colony_Mesh_To_SurfaceEvolverMesh
   SurfaceEvolver_Mesh_To_Force_Network
   Compare_force_network_CellFIT_DLITE
   Simulate_FOV_Drift

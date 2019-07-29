=====================
DLITE
=====================


.. image:: https://travis-ci.com/AllenCellModeling/DLITE.svg?branch=master
        :target: https://travis-ci.com/AllenCellModeling/DLITE
        :alt: Build Status

.. image:: https://readthedocs.org/projects/dlite/badge/?version=latest
        :target: https://DLITE.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://codecov.io/gh/AllenCellModeling/DLITE/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/AllenCellModeling/DLITE
  :alt: Codecov Status


DLITE is Dynamic Local Intercellular Tension Estimation of cell-cell forces in time-lapse images of a cell monolayer. In other words, DLITE is a method that takes in a digested skeletonization of curved edges at the tops of cell colonies and gives you a predicted tension along each of those edges. This differs from other tools such as `CellFIT`_ in that it is intended to be applied across time series; the predictions get better as you feed more and more frames into it. 

Full documentation is available here_.

.. _here: https://DLITE.readthedocs.io.
.. _CellFIT: http://www.civil.uwaterloo.ca/brodland/inferring-forces-in-cells.html 
   
.. image:: https://user-images.githubusercontent.com/40371793/61190376-3f3bf800-a650-11e9-9e8f-51235200aca4.jpg
   :width: 750px
   :scale: 100 %
   :align: center
  
   
.. Add a section on what DLITE can do, as bullet points (It can: - load such and such format -...)
.. Add a section on what DLITE needs as inputs, how the input files need to be formatted

Features
--------

| Predict tensions in Surface Evolver data


* Data is available as txt files (/Notebooks/data/voronoi_very_small_44_edges_tension_edges_20_30_1.0.fe.txt):


.. code-block:: bash

    $ (env)> cd Notebooks

* Run demo notebook :

.. code-block:: bash

    $ (env)> jupyter notebook demo_notebook_SurfaceEvolver.ipynb

| Predict tensions in ZO-1 data


* Data is available as txt files (/Notebooks/data/MAX_20170123_I01_003-Scene-4-P4-split_T0.ome.txt):


.. code-block:: bash

    $ (env)> cd Notebooks

* Run demo notebook :

.. code-block:: bash

    $ (env)> jupyter notebook demo_notebook_ZO-1.ipynb


Support
-------
We are not currently supporting this code, but simply releasing it to the community AS IS but are not able to provide any guarantees of support. The community is welcome to submit issues, but you should not expect an active response.

Additional
-------

* Licensed under the `Allen Institute Software License`_.
* This package was created with Cookiecutter_.

.. _Allen Institute Software License: https://github.com/AllenCellModeling/DLITE/blob/master/LICENSE
.. _Cookiecutter: https://github.com/audreyr/cookiecutter

=====================
DLITE
=====================


.. image:: https://img.shields.io/pypi/v/DLITE.svg
        :target: https://pypi.python.org/pypi/DLITE
        :alt: Release Status

.. image:: https://travis-ci.com/AllenCellModeling/DLITE.svg?branch=master
        :target: https://travis-ci.com/AllenCellModeling/DLITE
        :alt: Build Status

.. image:: https://readthedocs.org/projects/DLITE/badge/?version=latest
        :target: https://DLITE.readthedocs.io/en/latest
        :alt: Documentation Status

.. image:: https://codecov.io/gh/AllenCellModeling/DLITE/branch/master/graph/badge.svg
        :target: https://codecov.io/gh/AllenCellModeling/DLITE
        :alt: Codecov Status


Dynamic Local Intercellular Tension Estimation
of cell-cell forces in time-lapse images of a cell monolayer

.. image:: https://user-images.githubusercontent.com/40371793/53372871-d0f8d200-3908-11e9-93f0-b006af0a4cb0.jpg
   :height: 100px
   :width: 200px
   :scale: 100 %
   :align: center
   

* Free software: Allen Institute Software License

* Documentation: https://DLITE.readthedocs.io.


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

Credits
-------

This package was created with Cookiecutter_.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
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


Dynamic Local Intercellular Tension Estimation
of cell-cell forces in time-lapse images of a cell monolayer
   
.. image:: https://user-images.githubusercontent.com/40371793/61190376-3f3bf800-a650-11e9-9e8f-51235200aca4.jpg
   :width: 750px
   :scale: 100 %
   :align: center
  
   
* Free software: Allen Institute Software License

* Documentation: https://DLITE.readthedocs.io.

Tests
--------

* After forking the repo, run tox for pytest

.. code-block:: bash

    $ conda create --name DLITE python=3.7

* Run demo notebook :

.. code-block:: bash

    $ conda activate DLITE

.. code-block:: bash

    $ pip install -e .[all]

.. code-block:: bash

    $ tox

Features
--------

|| Predict tensions in synthetic data

* Data is available as txt files (/Notebooks/data/Synthetic_data/)

.. code-block:: bash

    $ cd Notebooks

* Run demo notebook :

.. code-block:: bash

    $ jupyter notebook demo_notebook_Surface_Evolver.ipynb

|| Predict tensions in ZO-1 data

* Data is available as txt files (/Notebooks/data/ZO-1_data/)

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

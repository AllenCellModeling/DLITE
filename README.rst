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
Organization
--------

The project has the following structure:

    DLITE/
      |- README.rst
      |- setup.py
      |- requirements.txt
      |- tox.ini
      |- Makefile
      |- MANIFEST.in
      |- HISTORY.rst
      |- CHANGES.rst
      |- AUTHORS.rst
      |- LICENSE
      |- docs/
         |- ...
      |- tests/
         |- __init__.py
         |- test_example.py
         |- data/
            |- ...
      |- DLITE/
         |- __init__.py
         |- cell_describe.py
         |- AICS_data.py
         |- ManualTracing.py
         |- ManualTracingMutliple.py
         |- SurfaceEvolver.py
         |- Lloyd_relaxation_class.py
         |- SaveSurfEvolFile.py
         |- PlottingFunctions.py
      |- Notebooks/
         |- Generate_Voronoi_Data.ipynb
         |- Demo_notebook_SurfaceEvolver.ipynb
         |- Demo_notebook_ZO-1.ipynb
         |- Compare_CELLFIT_DLITE.ipynb
         |- FOV_drift.ipynb
         |- Data/
            |- Synthetic_data
               |- ...
            |- ZO-1_data
               |- ...

Tests
--------

* After forking the repo, create a conda environment and run tests to confirm that required dependencies are installed

.. code-block:: bash

    $ conda create --name DLITE python=3.7

* Activate conda environment :

.. code-block:: bash

    $ conda activate DLITE

* Install requirments in setup.py

.. code-block:: bash

    $ pip install -e .[all]

* Run tests

.. code-block:: bash

    $ tox

Features
--------

**DLITE can**


* Generate synthetic colonies from relaxed voronoi tessellations and save them as .txt files in Surface Evolver format. 

  * Run demo notebook:

.. code-block:: bash

    $ jupyter notebook Generate_Voronoi_Data.ipynb

* Predict tensions in time-series synthetic data. Data is available as .txt files (/Notebooks/data/Synthetic_data/). 

  * Run demo notebook:

.. code-block:: bash

    $ jupyter notebook demo_notebook_Surface_Evolver.ipynb

* Predict tensions in time-series ZO-1 data. Data is available as .txt files (/Notebooks/data/ZO-1_data/). 

  * Run demo notebook:

.. code-block:: bash

    $ jupyter notebook demo_notebook_ZO-1.ipynb

* Compare tension predictions between CellFIT and DLITE. 

  * Run demo notebook:

.. code-block:: bash

    $ jupyter notebook Compare_CELLFIT_DLITE.ipynb

* Simulate field of view (FOV) drift within a single colony. 

  * Run demo notebook:

.. code-block:: bash

    $ jupyter notebook FOV_drift.ipynb

Usage
------

**DLITE needs**


* Input data in the form of .txt files. 
  * Synthetic data .txt files are formatted for the outputs of Surface Evolver.
  *  ZO-1 data .txt files are formatted for the outputs of manual tracing using the NeuronJ plugin in ImageJ.
 
Support
-------
We are not currently supporting this code, but simply releasing it to the community AS IS but are not able to provide any guarantees of support. The community is welcome to submit issues, but you should not expect an active response.

Additional
-------

* Licensed under the `Allen Institute Software License`_.
* This package was created with Cookiecutter_.

.. _Allen Institute Software License: https://github.com/AllenCellModeling/DLITE/blob/master/LICENSE
.. _Cookiecutter: https://github.com/audreyr/cookiecutter

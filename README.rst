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

`Install via pip`_ as described or view `full documentation`_. For more details, refer to our `biorxiv`_ paper.

.. _Install via pip: `Installation`_
.. _full documentation: https://DLITE.readthedocs.io/en/latest/?badge=latest
.. _CellFIT: http://www.civil.uwaterloo.ca/brodland/inferring-forces-in-cells.html 
.. _biorxiv: https://www.biorxiv.org/content/10.1101/541144v2.full
   
.. image:: https://user-images.githubusercontent.com/40371793/61190376-3f3bf800-a650-11e9-9e8f-51235200aca4.jpg
   :width: 750px
   :align: center
  
   
.. Add a section on what DLITE needs as inputs, how the input files need to be formatted


Installation 
------------

Like other complex projects, DLITE is best installed in a `Conda environment`_ or `virtual environment`_. We'll use a Conda environment here. First we'll create and activate it. 

.. code-block:: bash

    $ conda create --name dlite python=3.7
    $ conda activate dlite

Now we have our named environment. Next clone the DLITE repository and install the local copy

.. code-block:: bash

    $ git clone https://github.com/AllenCellModeling/DLITE.git
    $ cd DLITE
    $ pip install -e .[all]

DLITE is installed. Optionally we can check that the installation was successful by running a test suite by issuing the single command ``$ tox``. But likely of more interest is to look at the example notebooks via, e.g.:

.. code-block:: bash

    $ jupyter-notebook  Notebooks/Example.ipynb

This is the preferred method to install DLITE in its alpha form as it allows updates via:

.. code-block:: console

    $ git pull

.. _Conda environment: https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html
.. _virtual environment: https://docs.python.org/3/tutorial/venv.html

Features
--------

**DLITE can**


* Generate synthetic colonies from relaxed voronoi tessellations and save them as .txt files in Surface Evolver format. 

.. code-block:: bash

    $ jupyter notebook Generate_Voronoi_Data.ipynb

* Predict tensions in time-series synthetic data. Data is available as .txt files (/Notebooks/data/Synthetic_data/). 

.. code-block:: bash

    $ jupyter notebook demo_notebook_Surface_Evolver.ipynb

* Predict tensions in time-series ZO-1 data. Data is available as .txt files (/Notebooks/data/ZO-1_data/). 

.. code-block:: bash

    $ jupyter notebook demo_notebook_ZO-1.ipynb

* Compare tension predictions between CellFIT and DLITE. 

.. code-block:: bash

    $ jupyter notebook Compare_CELLFIT_DLITE.ipynb

* Simulate field of view (FOV) drift within a single colony. 

.. code-block:: bash

    $ jupyter notebook FOV_drift.ipynb

Usage
------

**DLITE needs**


* Input data in the form of .txt files. 

* Synthetic data .txt files that are formatted for the outputs of Surface Evolver.

*  Experimental data .txt files that are formatted for the outputs of tracing using the NeuronJ plugin in ImageJ.

Citation
--------

If you find this code useful in your research, please consider citing the following paper::

  @article{vasan2019dlite,
    title={DLITE uses cell-cell interface movement to better infer cell-cell tensions},
    author={Vasan, Ritvik and Maleckar, Mary M and Williams, Charles David and Rangamani, Padmini},
    journal={bioRxiv},
    pages={541144},
    year={2019},
    publisher={Cold Spring Harbor Laboratory}
  }

Support
-------
We are not currently supporting this code, but simply releasing it to the community AS IS but are not able to provide any guarantees of support. The community is welcome to submit issues, but you should not expect an active response.

Additional
----------

* Licensed under the `Allen Institute Software License`_.
* This package was created with Cookiecutter_.

.. _Allen Institute Software License: https://github.com/AllenCellModeling/DLITE/blob/master/LICENSE
.. _Cookiecutter: https://github.com/audreyr/cookiecutter

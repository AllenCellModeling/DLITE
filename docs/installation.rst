.. highlight:: shell

============
Installation
============


Updateable install
------------------

To install ``DLITE`` in an updatable and editable form, run these commands in your terminal:

.. code-block:: console

    $ conda create --name dlite
    $ conda activate dlite
    $ git clone git://github.com/AllenCellModeling/DLITE
    $ cd DLITE
    $ pip install -e .[all]

This is the preferred method to install ``DLITE`` in its current alpha form as it will allow updates via:

.. code-block:: console

    $ git pull

Direct pip install
------------------

It is also possible to install ``DLITE`` directly from the `Github repo`_ via:

.. code-block:: console

    $ pip install git+https://github.com/AllenCellModeling/DLITE.git

.. _Github repo: https://github.com/AllenCellModeling/DLITE
.. _tarball: https://github.com/AllenCellModeling/DLITE/tarball/master


# DLITE: Dynamic Local Intercellular Tension Estimation
Estimation of dynamic cell-cell forces from colony time-series. 

![fig7_2col](https://user-images.githubusercontent.com/40371793/53372871-d0f8d200-3908-11e9-93f0-b006af0a4cb0.jpg)

### Organization of the  project

The project has the following structure:

    DLITE/
      |- README.md
      |- cell_soap/
         |- __init__.py
         |- cell_describe.py
         |- AICS_data.py
         |- ManualTracing.py
         |- ManualTracingMutliple.py
         |- SurfaceEvolver.py
      |- Notebooks/
         |- Demo_notebook_SurfaceEvolver.ipynb
         |- Demo_notebook_ZO-1.ipynb
         |- data/
            |- ...
      |- setup.py
      |- CHANGES.txt
      |- MANIFEST.in
      |- .gitignore
      |- requirements.txt
      |- LICENSE

## Installation
### Environment setup
- Create a virtual environment:
```shell
python3 -m venv DLITE-env
```

- Activate the environment:
```shell
source DLITE-env/bin/activate
```

- Install requirements:
```shell
python setup.py install
```

## Predict tensions in Surface Evolver data

- Data is available as txt files (/Notebooks/data/voronoi_very_small_44_edges_tension_edges_20_30_1.0.fe.txt):
```shell
cd Notebooks
```

- Run demo notebook :
```shell
jupyter notebook demo_notebook_SurfaceEvolver.ipynb
```

## Predict tensions in ZO-1 data

- Data is available as txt files (/Notebooks/data/MAX_20170123_I01_003-Scene-4-P4-split_T0.ome.txt):
```shell
cd Notebooks
```

- Run demo notebook :
```shell
jupyter notebook demo_notebook_ZO-1.ipynb
```

## Contact
C. Dave Williams 
E-mail: <cdave@alleninstitute.org>

## License
Licensed under the Allen Institute Software License. See `LICENSE` for details. 

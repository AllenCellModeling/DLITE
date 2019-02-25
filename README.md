# DLITE: Dynamic Local Intercellular Tension Estimation
Estimation dynamic cell-cell forces from colony time-series. 

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
      |- Manifest.in
      |- LICENSE

## System Requirements

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

## Data
Data is available as txt files (/Notebooks/data). 

## Predict tensions in Surface Evolver data
Run demo_notebook_SurfaceEvolver.ipynb

## Predict tensions in ZO-1 data
Run demo_notebook_ZO-1.ipynb

## Contact
C. Dave Williams 
E-mail: <cdave@alleninstitute.org>

## Allen Institute Software License
Allen Institute Software License – This software license is the 2-clause BSD license plus clause a third clause that prohibits redistribution and use for commercial purposes without further permission.   
Copyright © 2018. Allen Institute.  All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.  
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.  
3. Redistributions and use for commercial purposes are not permitted without the Allen Institute’s written permission. For purposes of this license, commercial purposes are the incorporation of the Allen Institute's software into anything for which you will charge fees or other compensation or use of the software to perform a commercial service for a third party. Contact terms@alleninstitute.org for commercial licensing opportunities.  

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


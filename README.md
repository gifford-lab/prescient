[![PyPI version](https://badge.fury.io/py/prescient.svg)](https://badge.fury.io/py/prescient)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# prescient
Software for PRESCIENT (Potential eneRgy undErlying Single Cell gradIENTs), a generative model for modeling single-cell time-series.
+ Current paper version: https://www.biorxiv.org/content/10.1101/2020.08.26.269332v1
+ For paper pre-processing scripts, training bash scripts, pre-trained models, and visualization notebooks please visit https://github.com/gifford-lab/prescient-analysis.

## Documentation
Documentation is available at https://cgs.csail.mit.edu/prescient.

<!-- ![trajectories_gif](docs/assets/gifs/trajectories.gif) -->

## Requirements

+ pytorch 1.4.0
+ geomloss 0.2.3, pykeops 1.3
+ numpy, scipy, pandas, sklearn, tqdm, annoy
+ scanpy, pyreadr, anndata
+ Recommended: An Nvidia GPU with CUDA support for GPU acceleration (see paper for more details on computational resources)

## Bugs & Suggestions

Please report any bugs, problems, suggestions or requests as a [Github issue](https://github.com/gifford-lab/prescient/issues)

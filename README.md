# plrt-reservoir-operation_FS23
This repository contains the code used for the JoHX manuscript: "Deriving Reservoir Operational Groups using Piecewise Linear Regression Trees".

## Authors
Lucas Ford, Jiheun Kim, A. Sankarasubramanian

The study uses a Piecewise Linear Regression Tree (PLRT)-based framework to identify reservoir operational groups and simulate reservoir release behavior across multiple reservoirs using standardized hydrologic and storage variables.

This repository includes:
- a rule-based PLRT model for release prediction,
- simulation code for recursive reservoir operation prediction,
- optional boundary-constrained simulation,
- evaluation scripts for multi-frequency data assimilation experiments.

## Relation to PLRT

The PLRT implementation used in this study was developed based on the ideas introduced in Alexander and Grimshaw (1996), and follows the structure and concepts of the original `py-plrt` repository developed by Lucas Ford:

- Original repository: https://github.com/lcford2/py-plrt

The original package combines decision trees and linear regression to create Piecewise Linear Regression Trees (PLRTs), where each terminal node uses a linear regression model instead of a mean-only prediction. This improves interpretability while allowing the model to capture linear relationships within each partition of the feature space.

In this study, the PLRT concept was adapted into a rule-based reservoir operation model for recursive simulation of reservoir release and storage.

The full implementation and workflow are written in "plrt_FS23_simulation.ipynb". This notebook contains:
- the PLRT rule-based model,
- recursive simulation logic,
- boundary-constrained simulation,
- and performance evaluation.

## Data

The repository includes the required input files directly:

- `meta_all(508).csv`
- `key_all(508)_stodiff_fixed.csv`
- `std_total(508)_stodiff_fixed.zip`
- `raw_total(508)_stodiff_fixed.zip`

The standardized and raw reservoir datasets are provided as compressed `.zip` files.

Before running the notebook, you must extract them.

A subset of reservoir data (21 reservoirs) used in this study is derived from Li et al (2025), where the dataset is available via HydroShare:
https://www.hydroshare.org/resource/63add4d5826a4b21a6546c571bdece10/

Li, D., Chen, Y., Cai, X., Zhao, Q. (2025).  
*Data-driven Reservoir Operation Rules for 450+ Reservoirs in Contiguous United States*.  
HydroShare. https://doi.org/10.4211/hs.63add4d5826a4b21a6546c571bdece10





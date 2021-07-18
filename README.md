# selanneal

Selanneal is a simple package for optimising multivariate selections via a figure of merit.
The optimisation is performed for all given features simultaneously by utilising the simulated annealing method.
It relies on [numba](http://numba.pydata.org/) for just-in-time compilation of the algorithm.
The procedure works on binned data, so an n-dimensional histogram needs to be provided.

Currently, two modes of operation exist:
* **edges**: cut only the edges of each feature (results in "rectangular cuts")
* **bins**: select individual bins from a grid (for now limited to 2 feature dimensions)

This package was written for applications in high energy physics but can apply to general problems in statistical data analysis.

## usage

* install with
```console
python3 -m pip install selanneal
```
* tutorial notebooks for basic usage in *examples*
* training data is to be provided as numpy arrays representing the histogrammed number of signal and background events
* hyper-parameters to tune the optimisation: <img src="https://render.githubusercontent.com/render/math?math=T_{max},T_{min},N_{steps}">
* the implemented figure of merit is <img src="https://render.githubusercontent.com/render/math?math=\frac{N_{sig}}{\sqrt{N_{sig}%2bN_{bkg}}}">

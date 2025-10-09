# gala - Genetic Algorithm for Lattice Antenna

github repo for the code used in [this paper](https://doi.org/10.1371/journal.pcbi.1012845)
## hey man what the dickens does all this stuff do anyway

Broadly: it is a (hopefully fairly extensible and configurable) collection of bits which you can use to run genetic algorithms, specifically related to photosynthetic light harvesting.
The main bits of the code are:

- The genetic algorithm itself (located in `genetic_algorithm.py`, funnily enough). In here you define the genes in your genome, along with some metadata about them that tells the operators in the algorithm what to do with those genes.
- A module to set up light environments (`light.py` - I'm very imaginative) for your funny little theoretical photosynthesis guys. There are various options here, like different stellar illuminations, terrestrial AM1.5 sunlight, various filters you can apply, options to change the intensity of the light, and so on.
- A set of solvers (`solvers.py`!) which set up and solve the relevant equations. This is kinda the guts of the whole thing. Basically we assume that our genome contains some information about energy transfer rates, and then the solver uses those rates to set up and solve a matrix of equations to ultimately give you occupation probabilities of the different states in the system. This information is then used to calculate whatever outputs you need and ultimately to rank the fitness of all the funny little bugs the genetic algorithm is creating.
- A stats module (guess what it's called) that takes pandas dataframes created from the population of genomes and does stats on them. The implementation of this is a bit of a mess honestly, I think, but the genome is generally made up of various different genes of different variable types (integer, float, string, arrays thereof) and there are different levels of granularity we might want in the stats (a straight average across the population, an elementwise average of an array, a bar chart of categorical variables, etc.), so it's always gonna be a bit weird to look at.

Other than that there's some utilities in `utils.py` and various hyperparameters and stuff in `constants.py`. Hopefully everything is commented well enough that you can figure out what it does.

## how do i use all this stuff

`argparse` is set up both for `main.py` and `simulation.py`. Basically you hand-write or generate a list of dicts, where each dict represents an input spectrum. Then `main.py` will call `light.py` to build the actual spectra from those specifications, generate the directory structures and save the spectrum files, and then optionally call `simulation.py` to actually do the simulations, one per spectrum. This is maybe not the most obvious or intuitive way of doing things, but I needed a way to set it up so I could submit batches of parallel jobs on a cluster, and this was the way that required the least refactoring. There are several options for `main.py` and `simulation.py` which you can check with the `-h` flag; other hyperparameters etc. are in `constants.py` as mentioned above.

## Requirements

### Python
You need Python, obviously. I have not tested the code on different versions of python with different packages so if you have some insane setup like you're still running python 2 or something please do not bother emailing me. I think any version > 3.9 should do (but see below for one potential issue if you do have an older version).

If in doubt, one thing I have tested is getting it running from a fresh miniforge install: simply install [miniforge](https://conda-forge.org/download/) and then do
```
conda install cython numpy scipy pandas matplotlib seaborn setuptools mamba
python setup.py build_ext --inplace
```
to build the cython module called by the solvers. This is currently all Python 3.14, I think.

### other

I also have a modern fortran version of scipy's NNLS solver included here; it's not currently used in the code but you can turn it on by passing
`solver_kwargs={'method': 'nnls'}`
to one of the solvers if you really want to.
If you do want to do this you'll need a fortran compiler to build the library.
There's a makefile included and any modern compiler should do, it's strict F2008.

### issues i've run into

- before moving to miniforge and updating to python 3.14 I was using a fairly old version of Python, i think 3.10. if you happen to be doing this as well (inadvisable in 2025, i know), you might need to add the option
`define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]`
after the `include_dirs` line in `setup.py` in order to get the cython to compile. this is because your version of numpy will be old and at some point (unsure when, can't be bothered to find out) some internal change was made to the pyarray structs, the class members are different and it'll throw an error.

- You might also need Qt by default for matplotlib; I'm not sure but I think the default renderer is QtAgg. if you get an error that looks something like
`qt.qpa.plugin: Could not find the Qt platform plugin .....`
then there's some kind of Qt error. setting `backend` to something else (i use `pdf`) in your `matplotlibrc` will fix this, or just add `--no-stats` when you run the code and it won't do any plots. i'd assume this is more likely if you're running the code through WSL or on an HPC cluster like i am because you're less likely to have gui stuff set up.

## TO DO:

- Try to standardise the output of all the stats functions, as much as possible at least. Requires some thought
- lookup tables for photon input rates and overlap between adjacent subunits. in theory this is mostly done but i haven't tested it yet
- some more intuitive way of setting up the initial spectra would be good, but i need to think about how that will work

# gala - Genetic Algorithm for Lattice Antenna

github repo for the code used in [this paper](https://doi.org/10.1371/journal.pcbi.1012845)

## hey man what the dickens does all this stuff do anyway

Broadly: it is a (hopefully fairly extensible and configurable) collection of bits which you can use to run genetic algorithms, specifically related to photosynthetic light harvesting.
The main bits of the code are:

- The genetic algorithm itself (located in `genetic_algorithm.py`, funnily enough). In here you define the genes in your genome, along with some metadata about them that tells the operators in the algorithm what to do with those genes.
- A module to set up light environments (`light.py` - I'm very imaginative) for your funny little theoretical photosynthesis guys. There are various options here, like different stellar illuminations, terrestrial AM1.5 sunlight, various filters you can apply, options to change the intensity of the light, and so on.
- A set of solvers (`antenna.py`) which set up and solve the relevant equations. This is kinda the guts of the whole thing. Basically we assume that our genome contains some information about energy transfer rates, and then the solver uses those rates to set up and solve a matrix of equations to ultimately give you occupation probabilities of the different states in the system. This information is then used to calculate whatever outputs you need and ultimately to rank the fitness of all the funny little bugs the genetic algorithm is creating.
- A stats module (guess what it's called) that takes the populations of genomes and calulcates stats for them. The implementation of this is a bit of a mess honestly, I think, but the genome is generally made up of various different genes of different variable types (integer, float, string, arrays thereof) and there are different levels of granularity we might want in the stats (a straight average across the population, an elementwise average of an array, a bar chart of categorical variables, etc.), so it's always gonna be a bit weird to look at.

Other than that there's various hyperparameters and stuff in `constants.py` and various plotting scripts and so forth. Hopefully everything is commented well enough that you can figure out what it does.

## how do i use all this stuff

Firstly you need to set everything up: in Linux or on a Mac or on Windows via WSL, make sure you have some sort of conda installed (i use conda-forge from [here](https://conda-forge.org/download/)). Download the Mac version if you're on a Mac or the Linux version for Linux/WSL, install it from your terminal, clone this repo, then do

`conda create -f environment.yml`
or
`mamba create -f environment.yml`

in the `gala` directory and it should install all the python packages you need. In theory if you can get a Python distribution and a Fortran compiler working on Windows you should also be able to run everything but I haven't tried and I will not figure it out for you because i hate windows with my life.

Once that's done: see `main.py` and `light.py` for the details but basically you set up the light environments you want to simulate, fix a value of the cost parameter, and then the code will run itself for those parameters (plus some other hyperparameters that control things like the reproduction algorithm, how new genomes are generated and so on). Once everything is set up as you want just run `python main.py` and it should all run for you. The python should compile my fortran NNLS solver for you and the output files should be put in sensibly-named directories which by default will be in the folder `out`.

## Requirements

### Python
You need Python, obviously. I have not tested the code on different versions of python with different package versions or anything; so if you have some insane setup, like you're still running python 2 or something, please do not bother emailing me. I think any version > 3.9 should do (but see below for a couple of potential issues if you do have a very old version).

If in doubt, one thing I have tested is getting it running from a fresh miniforge install: simply install [miniforge](https://conda-forge.org/download/) and then do
```
conda install cython numpy scipy pandas matplotlib seaborn setuptools
python setup.py build_ext --inplace
```
to build the cython module called by the solvers. This is currently all Python 3.14, I think.

### other

I also have a modern fortran version of scipy's NNLS solver included here; it's not currently used in the code but you can turn it on by passing
`solver_kwargs={'method': 'nnls', 'nnls_method': 'fortran'}`
to one of the solvers if you really want to.
If you do want to do this you'll need a fortran compiler to build the library.
There's a makefile included and any modern compiler should do, it's strict F2008.

### issues i've run into

- before moving to miniforge and updating to python 3.14 I was using a fairly old version of Python, i think 3.10. if you happen to be doing this as well (inadvisable but i see you, i too am too lazy to update anything), you might need to add the option
`define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]`
after the `include_dirs` line in `setup.py` in order to get the cython to compile. this is because your version of numpy will also be old and at some point (unsure when, can't be bothered to find out) some internal change was made to the pyarray structs-- the class members have changed and the compiler will throw an error without those macros.
- if you get an `argparse` error that says something about `BooleanOptionalAction` you're using an ancient version of python (ancient here meaning >7 years and no longer supported) and you desperately need to update. i only found this out because i tried to run some simulations on an old machine with an old anaconda install on it and the base version was 3.7.
- You might also need Qt by default for matplotlib; I'm not sure but I think the default renderer is QtAgg. if you get an error that looks something like
`qt.qpa.plugin: Could not find the Qt platform plugin .....`
then (obviously) there's some kind of Qt error. explicitly setting `backend` to something non-Qt-related (i use `pdf`) in `matplotlibrc` fixed this for me, or just add `--no-stats` when you run the code and it won't make any plots. i'd assume this is more likely if you're running the code through WSL or on an HPC cluster like i am because you're less likely to have gui stuff set up.

## TO DO:

- Try to standardise the output of all the stats functions, as much as possible at least. Requires some thought
- lookup tables for photon input rates and overlap between adjacent subunits. in theory this is mostly done but i haven't tested it yet
- some more intuitive way of setting up the initial spectra would be good, but i need to think about how that will work

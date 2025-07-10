# gala - Genetic Algorithm for Lattice Antenna

github repo for the code used in [this paper](https://doi.org/10.1371/journal.pcbi.1012845)

## Requirements

- a modern Python version, I think 3.9 or above
- various standard packages, numpy, matplotlib, pandas, seaborn, etc.
- optionally a fortran compiler to compile my fortran version of scipy's NNLS. any gfortran will do, it is strict F2008

## hey man what the dickens does all this stuff do anyway

Broadly: it is a (hopefully fairly extensible and configurable) collection of bits which you can use to run genetic algorithms, specifically related to photosynthetic light harvesting.
The main bits of the code are:

- The genetic algorithm itself (located in `genetic_algorithm.py`, funnily enough). In here you define the genes in your genome, along with some metadata about them that tells the operators in the algorithm what to do with those genes.
- A module to set up light environments (`light.py` - I'm very imaginative) for your funny little theoretical photosynthesis guys. There are various options here, like different stellar illuminations, terrestrial AM1.5 sunlight, various filters you can apply, options to change the intensity of the light, and so on.
- A set of solvers (`solvers.py`!) which set up and solve the relevant equations. This is kinda the guts of the whole thing. Basically we assume that our genome contains some information about energy transfer rates, and then the solver uses those rates to set up and solve a matrix of equations to ultimately give you occupation probabilities of the different states in the system. This information is then used to calculate whatever outputs you need and ultimately to rank the fitness of all the funny little bugs the genetic algorithm is creating.
- A stats module (guess what it's called) that takes pandas dataframes created from the population of genomes and does stats on them. The implementation of this is a bit of a mess honestly, I think, but the genome is generally made up of various different genes of different variable types (integer, float, string, arrays thereof) and there are different levels of granularity we might want in the stats (a straight average across the population, an elementwise average of an array, a bar chart of categorical variables, etc.), so it's always gonna be a bit weird to look at.

Other than that there's some utilities in `utils.py` and various hyperparameters and stuff in `constants.py`. Hopefully everything is commented well enough that you can figure out what it does.

## how do i use all this stuff

I haven't sorted out any argparse business or anything so all the setup is done in `main.py` basically. i set up a dict of parameters for the different spectra i want to use, set a value for the cost parameter (which basically controls antenna size so it doesn't go too crazy) and then do `python main.py` in a terminal to run it. See `main.py` for the details and the rest of the python files for comments on all the functions that are called.

## TO DO:

- Try to standardise the output of all the stats functions, as much as possible at least. Requires some thought
- Figure out how to reduce computation time more, by using more lookup tables in `utils.py`. My ultimate plan is that for a given genome and light environment, I start by pre-calculating overlaps and photon input rates for every possible combination, and then as we go add the output of the solver to a big database or something so that if an identical genome is generated later, we can just look up the result instead of having to recalculate.

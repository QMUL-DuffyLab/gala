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

## extending it

### pigments

Pigment details are in the file `pigments/pigment_data.json`. What I've done is to take pigments of interest, fit them to a set of Gaussians (makes it easier to shift their absorption if you want to do that, and also avoids having to do any work to make sure the wavelength values are consistent with the input spectrum) and then the `bounds` dict in `constants.py` contains a list of all the pigments that the genetic algorithm will use. The keys in that list must match those in the JSON file, and you'll need to fit both absorption and emission spectra. Hopefully the syntax in the JSON file is self-explanatory.

### light inputs

There are a couple of wrapper functions in `light.py` that then call a few other functions to (for example) filter out blue light, pull up a PHOENIX stellar spectrum, or whatever else. All of these internal functions return a 2-column array of x (wavelength) and y (irradiance), and then the wrapper function `spectrum_setup` also generates an output prefix which will be prefixed to the output files to make them easier to keep track of. So to add new light environments you can add a function in `light.py` to generate the array and then add them to that wrapper function, or just read in a text file and make an output prefix yourself.

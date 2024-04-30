import numpy as np
import constants
import genetic_algorithm as ga
# import pyjulia
# from julia import DrawAntennae

def pigment_to_index(pigment):
    ''' 
    convert the pigment name to an index for histograms.
    the [0][0] is because where returns a tuple with
    an array of locations as the first element; since each name
    only occurs once in bounds['pigment'] we can take the first
    element of this array safely. looks stupid though.
    Note - this breaks if pigment is not in bounds['pigment'],
    so I do that check manually below
    '''
    return np.where(constants.bounds['pigment'] == pigment)[0][0]

def hist(population, gen, run, outdir, out_name):
    suffix = "hist_{}_{:04d}_{:1d}.dat".format(out_name, gen, run)
    l_bin_size = 1.0
    s_max = constants.hist_sub_max
    n_pop = constants.population_size
    # this needs changing to be e.g. 200-1400, now i've changed lp
    l_b = constants.bounds['lp']
    lbins = np.linspace(*l_b,
            num=np.round(((l_b[1] - l_b[0]) / l_bin_size)).astype(int))
    pvals = constants.bounds['pigment']
    l_arr = np.zeros((s_max, n_pop), dtype=np.float64)
    p_arr = np.zeros((s_max, n_pop), dtype='U10')
    '''
    for the first hist_sub_max subunits, make a histogram
    of peak wavelength and a count of pigment types,
    normalised by total population
    '''
    for j, p in enumerate(population):
        for i in range(s_max):
            if i < p.n_s:
                par = constants.pigment_data[p.pigment[i]]
                # p.lp[i] is the offset now!
                l_arr[i][j] = p.lp[i] + par['lp'][0]
                p_arr[i][j] = p.pigment[i]
    lh = [lbins[:-1]] # len(lbins) = len(hist) + 1
    ph = [pvals]
    for i in range(s_max):
        # [0] is the histogram, otherwise it adds the bins too
        hist = np.histogram(l_arr[i], bins=lbins)[0]
        lh.append(hist / n_pop)
        pcount = np.zeros(len(pvals), dtype=np.float64)
        for j in range(n_pop):
            if p_arr[i][j] != '':
                pcount[pigment_to_index(p_arr[i][j])] += 1
        ph.append(pcount / n_pop)
    lfile = outdir + "/" + "lp_" + suffix
    pfile = outdir + "/" + "pigment_" + suffix
    np.savetxt(lfile, np.transpose(np.array(lh)))
    np.savetxt(pfile,
               np.transpose(np.array(ph, dtype=object)),
               fmt="%s" + s_max * " %.18e")
    return pfile, lfile

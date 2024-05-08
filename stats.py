import numpy as np
import constants
import genetic_algorithm as ga

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
    s_max = constants.hist_sub_max
    n_pop = constants.population_size
    # peak wavelength histogram with offset
    peak_b = [400.0, 800.0]
    peak_binsize = 1.0
    peakbins = np.linspace(*peak_b,
            num=np.round(((peak_b[1] - peak_b[0]) / peak_binsize)).astype(int))
    peak_arr = np.zeros((s_max, n_pop), dtype=np.float64)

    # n_p histogram
    n_p_arr = np.zeros((s_max, n_pop), dtype=np.int64)

    # pigment histogram
    pvals = constants.bounds['pigment']
    p_arr = np.zeros((s_max, n_pop), dtype='U10')
    '''
    for the first hist_sub_max subunits, make a histogram
    of peak wavelength, number of pigments and a count of pigment types,
    normalised by total population
    '''
    for j, p in enumerate(population):
        for i in range(s_max):
            if i < p.n_s:
                par = constants.pigment_data[p.pigment[i]]
                # p.lp[i] is the peakset now!
                peak_arr[i][j] = p.lp[i] + par['lp'][0]
                p_arr[i][j] = p.pigment[i]
                n_p_arr[i][j] = p.n_p[i]
    peakh = [peakbins[:-1]] # len(peakbins) = len(hist) + 1
    nph = []
    ph = [pvals]
    for i in range(s_max):
        # [0] is the histogram, otherwise it adds the bins too
        hist = np.histogram(peak_arr[i], bins=peakbins)[0]
        peakh.append(np.copy(hist) / n_pop)

        # minlength = upper bound on n_p + 1 because 0 is also possible
        hist = np.bincount(n_p_arr[i],
                           minlength=constants.bounds['n_p'][1] + 1)
        nph.append(np.copy(hist) / n_pop)
        
        pcount = np.zeros(len(pvals), dtype=np.float64)
        for j in range(n_pop):
            if p_arr[i][j] != '':
                pcount[pigment_to_index(p_arr[i][j])] += 1
        ph.append(pcount / n_pop)

    peak_file = outdir + "/" + "peak_" + suffix
    n_p_file = outdir + "/" + "n_p_" + suffix
    pfile = outdir + "/" + "pigment_" + suffix
    np.savetxt(peak_file, np.transpose(np.array(peakh)))
    np.savetxt(n_p_file,
               np.column_stack((np.arange(0, constants.bounds['n_p'][1] + 1),
                                np.transpose(np.array(nph)))))
    np.savetxt(pfile,
               np.transpose(np.array(ph, dtype=object)),
               fmt="%s" + s_max * " %.18e")
    return pfile, peak_file, n_p_file

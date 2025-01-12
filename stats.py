import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import constants
import plots
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

def get_average_vals_errs(basepath, cost, out_name):
    '''
    pull out the values and errors from the averaging procedure
    for a given set of parameters. note that this is all obviously
    dependent on the directories being set up as expected and the
    formatting in the averages file being as below, otherwise the
    regexes won't pick anything up. check for nans in the returned
    arrays
    '''
    fn = os.path.join(basepath,
            f"cost_{cost}", f"{out_name}_average_antenna_params.dat")
    vals = np.zeros(5, dtype=np.float64)
    errs = np.zeros_like(vals)
    try:
        with open(fn, "r") as f:
            fstr = ' '.join(f.read().splitlines())
        m = re.search(r'<nu_e> = ([0-9.]+) \+\= ([0-9.]+)', fstr)
        vals[0] = float(m.group(1))
        errs[0] = float(m.group(2))
        m = re.search(r'<phi_e> = ([0-9.]+) \+\= ([0-9.]+)', fstr)
        vals[1] = float(m.group(1))
        errs[1] = float(m.group(2))
        m = re.search(r'<fit> = ([0-9.]+) \+\= ([0-9.]+)', fstr)
        vals[2] = float(m.group(1))
        errs[2] = float(m.group(2))
        m = re.search(r'n_b = ([0-9.]+) \+\= ([0-9.]+)', fstr)
        vals[3] = float(m.group(1))
        errs[3] = float(m.group(2))
        m = re.search(r'n_s = ([0-9.]+) \+\= ([0-9.]+)', fstr)
        vals[4] = float(m.group(1))
        errs[4] = float(m.group(2))
    except:
        vals = np.nan
        errs = np.nan
    return vals, errs

def hist(population, gen, run, outdir, out_name):
    suffix = "hist_{}_{:04d}_{:1d}.dat".format(out_name, gen, run)
    s_max = constants.hist_sub_max
    n_pop = constants.population_size
    # peak wavelength histogram with offset
    peak_b = constants.x_lim
    peakbins = np.linspace(*peak_b,
            num=np.round(((peak_b[1] - peak_b[0])
                / constants.peak_binwidth)).astype(int))
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
                # p.shift[i] shifts the 0-0 lines
                peak_arr[i][j] = p.shift[i] * constants.shift_inc + par['abs']['mu'][0]
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

    peak_file = os.path.join(outdir, f"peak_{suffix}")
    n_p_file = os.path.join(outdir, f"n_p_{suffix}")
    pfile = os.path.join(outdir, f"pigment_{suffix}")
    np.savetxt(peak_file, np.transpose(np.array(peakh)))
    np.savetxt(n_p_file,
               np.column_stack((np.arange(0, constants.bounds['n_p'][1] + 1),
                                np.transpose(np.array(nph)))))
    np.savetxt(pfile,
               np.transpose(np.array(ph, dtype=object)),
               fmt="%s" + s_max * " %.18e")
    return pfile, peak_file, n_p_file

def average_antenna(path, spectrum, out_name):
    '''
    averages - take floats for n_b/n_s, sum and average, then round
    for n_p and pigments, sum the histograms, then take the mode of the result? 
    then divide n_p through and round to int
    '''
    totals = np.zeros(5)
    errors = np.zeros(5)
    n_ps = []
    peaks = []
    pigments = []
    avg_spectrum = np.zeros_like(spectrum)
    for i in range(constants.n_runs):
        avg_file = os.path.join(path, "avg_{}_r{:1d}.dat".format(out_name, i))
        avg_sq_file = os.path.join(path, "avgsq_{}_r{:1d}.dat".format(out_name, i))
        with open(avg_file, "r") as f:
            for line in f:
                pass
            final = np.fromstring(line, sep=' ')
        totals[0] += final[0]
        totals[1] += final[2]
        totals[2] += final[3]
        totals[3] += final[6]
        print(f"n_b = {final[6]}, n_s = {final[7]}")
        totals[4] += final[7]
        with open(avg_sq_file, "r") as f:
            for line in f:
                pass
            final = np.fromstring(line, sep=' ')
        errors[0] += final[0] / constants.population_size
        errors[1] += final[2] / constants.population_size
        errors[2] += final[3] / constants.population_size
        errors[3] += final[6] / constants.population_size
        errors[4] += final[7] / constants.population_size
        # now we need the highest numbered pigment and n_p histograms surely i can think of a way to do this. should've made a log file
        n_p_histfiles = glob.glob(os.path.join(path, "n_p_hist_{}_*_{:1d}.dat".format(out_name, i)))
        pigment_histfiles = glob.glob(os.path.join(path, "pigment_hist_{}_*_{:1d}.dat".format(out_name, i)))
        peak_histfiles = glob.glob(os.path.join(path, "peak_hist_{}_*_{:1d}.dat".format(out_name, i)))
        peak_file = sorted(peak_histfiles)[-1]
        n_p_file = sorted(n_p_histfiles)[-1]
        pigment_file = sorted(pigment_histfiles)[-1]
        peak_bins = np.loadtxt(peak_file, usecols=0)
        peaks.append(np.loadtxt(peak_file, usecols=range(1, constants.hist_sub_max + 1)))
        n_ps.append(np.loadtxt(n_p_file, usecols=range(1, constants.hist_sub_max + 1)))
        pigment_names = np.loadtxt(pigment_file, usecols=0, ndmin=1, dtype=str)
        pigments.append(np.loadtxt(pigment_file, usecols=range(1, constants.hist_sub_max + 1)))
        a = np.loadtxt(os.path.join(path, f"avg_{out_name}_r{i:1d}_spectrum.dat"))
        avg_spectrum += a
        # now we need the mode from each column of n_ps and pigment_props; index pigment_props back into pigment_names to get the right one

    errors = np.sqrt(errors)
    avg_spectrum /= constants.n_runs
    totals /= constants.n_runs
    print(f"<nu_e> = {totals[0]} += {errors[0]}")
    print(f"<phi_e> = {totals[1]} += {errors[1]}")
    print(f"<fit> = {totals[2]} += {errors[2]}")
    print(f"<n_b> = {totals[3]} += {errors[3]}")
    print(f"<n_s> = {totals[4]} += {errors[4]}")
    
    n_p_hist_avg = np.sum(np.array(n_ps), axis=0) / constants.n_runs
    peak_hist_sum = np.sum(np.array(peaks), axis=0) / constants.n_runs
    n_p_avg = np.zeros(constants.hist_sub_max)
    for i, col in enumerate(np.transpose(n_p_hist_avg)):
        n_p_avg[i] = np.sum([(j * col[j]) for j in range(len(col))])

    peak_avg = np.zeros(constants.hist_sub_max)
    for i, col in enumerate(np.transpose(peak_hist_sum)):
        peak_avg[i] = np.sum([(peak_bins[j] * col[j]) for j in range(len(col))])

    # don't actually need to divide through pigments - just taking mode
    pigments_avg = np.sum(np.array(pigments), axis=0)
    pigments_final = []
    fstr = "['{:s}'" + constants.hist_sub_max * ', {:10.6e}' + '],'
    average_antenna_output_file = os.path.join(path, f"{out_name}_average_antenna_params.dat")
    with open(average_antenna_output_file, "w") as f:
        f.write(f"<nu_e> = {totals[0]} += {errors[0]}\n")
        f.write(f"<phi_e> = {totals[1]} += {errors[1]}\n")
        f.write(f"<fit> = {totals[2]} += {errors[2]}\n")
        f.write(f"n_b = {totals[3]} += {errors[3]}\n")
        f.write(f"n_s = {totals[4]} += {errors[4]}\n")
        f.write(f"<n_p> = {n_p_avg}\n")
        f.write("\n")
        f.write(f"peaks: {peak_avg}\n")
        if len(pigment_names) > 1:
            f.write("pigments = ")
            f.write("[")
            for i in range(len(pigment_names)):
                row = [pigment_names[i]] + list(pigments_avg[i, :])
                print(row)
                pigments_final.append(row)
                f.write(fstr.format(*row))
            f.write("]")

    fig, ax = plt.subplots(figsize=(9,9))
    plt.plot(spectrum[:, 0], spectrum[:, 1], label='Incident',
            color=plots.get_spectrum_colour(out_name), lw=5.0)
    x = avg_spectrum[:, 0]
    y = avg_spectrum[:, 1]/np.sum(avg_spectrum[:, 1])
    np.savetxt(os.path.join(path, f"{out_name}_avg_avg_spectrum.dat"),
            np.column_stack((x, y)))
    plt.plot(avg_spectrum[:, 0], avg_spectrum[:, 1] / np.max(avg_spectrum[:, 1]),
            color='k', lw=5.0, label=r'$ \left<A\left(\lambda\right)\right> $')
    ax.set_xlabel(r'wavelength (nm)')
    ax.set_ylabel("intensity (arb.)")
    ax.set_xlim(constants.x_lim)
    plt.grid(True)
    # plt.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(path, f"{out_name}_avg_avg_spectrum.pdf"))
    plt.close()

    pigment_bar_file = os.path.join(path, f"{out_name}_pigment_bar.pdf")
    plots.pigment_bar(pigments_final, pigment_bar_file)

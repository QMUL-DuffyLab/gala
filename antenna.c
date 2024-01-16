#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <math.h>
#include <gsl/gsl_linalg.h>

/* various constants */

#if ! defined(M_PI)
	#define M_PI 3.1415926535897932384626433832795L
#endif

#if ! defined(M_C)
	#define M_C 299792458U
#endif

#if ! defined(M_H)
	#define M_H 6.62607015E-34L
#endif

#if ! defined(M_KB)
	#define M_KB 1.380649E-23L
#endif

struct pigment {
  unsigned n_gauss;
  double* lp;
  double* w;
  double* amp;
  char name[10];
};

typedef struct pigment Pigment;

Pigment
get_pigment_data(char* filename, char* pigment_name)
{
  /* NB: need an EOF check here too */
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    printf("pigment file not found\n");
    printf(filename);
    raise(SIGABRT);
  }
  char line[1024];
  char *data = line; /* can't increment line using offset */
  int offset; /* so we can read successive array elements */
  int ret;
  int debug = 0;
  char name[10] = "\0";
  unsigned n_gauss;
  while (strcmp(name, pigment_name)) {
    fgets(line, 1024, fp);
    ret = sscanf(line, "%s %u%n", name, &n_gauss, &offset);
    if (debug) {
      printf(line, ret, name, n_gauss, offset);
    }
  }
  data += offset;

  Pigment p;
  strcpy(p.name, name);
  p.n_gauss = n_gauss;
  /* these need to be freed in main once the lineshape's calculated */
  p.lp = calloc(n_gauss, sizeof(double));
  p.w = calloc(n_gauss, sizeof(double));
  p.amp = calloc(n_gauss, sizeof(double));
  for (unsigned j = 0; j < n_gauss; j++) {
    sscanf(data, "%le %le %le%n", &p.amp[j], &p.lp[j], &p.w[j], &offset);
    data += offset; /* move forward to the next array elements */
    if (j > 0) {
      /* we want the differences in peak wavelength because we
       * overwrite the reddest peak - that's what mutation does */
      p.lp[j] -= p.lp[0];
    }
  }
  if (debug) {
    printf("input pigment name: %s\n", pigment_name);
    printf("read pigment name: %s\n", p.name);
    printf("n_gauss: %d\n", p.n_gauss);
    for (unsigned j = 0; j < n_gauss; j++) {
      printf("amp, peak, width: %lf, %lf, %lf\n", p.amp[j], p.lp[j], p.w[j]);
    }
  }
  fclose(fp);
  return p;
}

double
trapezoid(double *f, double *x, unsigned n)
{
  /* trapezoid rule for f(x) with steps x */
  double result = 0.0;
  for (unsigned i = 1; i < n; i++) {
    result += (f[i - 1] + f[i]) * 0.5 * (x[i] - x[i - 1]);
  }
  return result;
}

double
overlap(double *l, double *f1, double *f2, unsigned n)
{
  /* calculate the overlap integral of (f1 * f2) with l */
  double *work = calloc(n, sizeof(double));
  for (unsigned i = 0; i < n; i++) {
    work[i] = f1[i] * f2[i];
  }
  double result = trapezoid(work, l, n);
  free(work);
  return result;
}

void
gauss(double *out, double *l, double lambda_peak, double width, unsigned n)
{
  /* normalised gaussian absorption lineshape */
  for (unsigned i = 0; i < n; i++) {
    out[i] = exp(-1.0 * pow(l[i] - lambda_peak, 2)/(2.0 * pow(width, 2)));
  }
  double norm = trapezoid(out, l, n);
  for (unsigned i = 0; i < n; i++) {
    out[i] /= norm;
  }
}

void
multi_gauss(double *out, double *l, unsigned l_len,
    unsigned n_gauss, double *lp, double *w, double *amp)
{
  /* NB: out must be zeroed before this! */
  for (unsigned j = 0; j < n_gauss; j++) {
    for (unsigned i = 0; i < l_len; i++) {
      out[i] += amp[j] * exp(-1.0 * pow(l[i] - lp[j], 2)/(2.0 * pow(w[j], 2)));
    }
  }
  double norm = trapezoid(out, l, l_len);
  for (unsigned i = 0; i < l_len; i++) {
    out[i] /= norm;
  }
}

double
dG(double l1, double l2, double n, double t)
{
  /* change in Gibbs free energy on moving from domain 1 -> 2.
   * enthalpy h, entropy s. note that g_21 = - g_12 so only do it once */
  double h = ((M_H * M_C) / 1.0E-9) * ((l1 - l2) / (l1 * l2));
  double s = -1.0 * M_KB * log(n);
  return (h - (t * s)); 
}

void
antenna(double *l, double *ip_y, double sigma, double k_params[5],
            double t, unsigned *n_p, double *lp, char **names,
            unsigned n_b, unsigned n_s, unsigned l_len,
            double* n_eq, double* nu_phi, int plot_lines)
{
  /* 
   * length of n_p, lp and names should be n_s + 1; RC params are in [0].
   * k_params should be given as [k_diss, k_trap,  k_con, k_hop, k_lhc_rc].
   * l_len is the length of the l and ip_y arrays
   * n_eq should be a vector with dim = (side).
   * nu_phi should be double[3].
   */
  unsigned side = (n_b * n_s) + 2;

  char pigment_file[] = "pigments/pigment_data.csv";

  gsl_vector *gamma = gsl_vector_calloc(side);
  gsl_permutation *perm = gsl_permutation_calloc(side);
  gsl_vector *n_eq_gsl = gsl_vector_calloc(side);
  int signum;

  double*  k_b   = calloc(2 * n_s, sizeof(double));
  double*  g     = calloc(n_s, sizeof(double));
  double*  fp_y  = calloc(l_len, sizeof(double));
  double** lines = calloc(n_s + 1, sizeof(double*));
  double** twa   = calloc(side, sizeof(double*));
  for (unsigned i = 0; i < side; i++) {
    twa[i]   = calloc(side,  sizeof(double));
    if (i < n_s + 1) {
      lines[i] = calloc(l_len, sizeof(double));
    }
  }

  /* absorption rates */
  for (unsigned i = 0; i < l_len; i++) {
    fp_y[i] = ip_y[i] * l[i] * ((1.0E-9)/(M_H * M_C));
  }
  
  /* calculate lineshapes */
  for (unsigned i = 0; i < n_s + 1; i++) {
    Pigment pigment = get_pigment_data(pigment_file, names[i]);
    pigment.lp[0] = lp[i];
    for (unsigned j = 1; j < pigment.n_gauss; j++) {
      pigment.lp[j] += pigment.lp[0];
    }
    multi_gauss(lines[i], l, l_len, pigment.n_gauss,
        pigment.lp, pigment.w, pigment.amp);
    if (plot_lines) {
      char fmt[] = "out/lineshapes/lineshape_%4d_%2d.dat";
      int sz = snprintf(NULL, 0, fmt, plot_lines, i);
      char buf[sz + 1];
      snprintf(buf, sizeof(buf), fmt, plot_lines, i);
      FILE *fp = fopen(buf, "w");
      if (fp) {
        for (unsigned j = 0; j < l_len; j++) {
          fprintf(fp, "%18.10e %18.10e\n", l[j], lines[i][j]);
        }
      }
      fclose(fp);
    }
    free(pigment.lp);
    free(pigment.w);
    free(pigment.amp);
    if ((i > 0) && (i < n_s + 1)) { 
      /* 
       * add to the vector of photon inputs for later.
       * note that n_p[0] and lines[0] are for the RC, so we skip that,
       * then subtract 1 to get back to correct indexing on g.
       */
      g[i - 1] = n_p[i] * sigma * overlap(l, fp_y, lines[i], l_len);
      for (unsigned j = 2; j < side; j = j + n_s) {
        /* j is the set of start indices for each branch */
        gsl_vector_set(gamma, i - 1 + j, -1.0 * g[i - 1]);
      }
    }
  }

  /* rate constants */
  for (unsigned i = 0; i < n_s; i++) {
    double de = overlap(l, lines[i], lines[i + 1], l_len);
    double n_ratio = (double)(n_p[i]) / (double)(n_p[i + 1]);
    /* free energy change moving *outward* */
    double delta_g = dG(lp[i], lp[i + 1], n_ratio, t);
    double rate = 0.0;
    if (i == 0) {
      rate = k_params[4]; /* k_LHC_RC */
    } else {
      rate = k_params[3]; /* k_hop */
    }
    rate *= de;
    k_b[(2 * i)]     = rate; /* outward */
    k_b[(2 * i) + 1] = rate; /* inward */
    if (delta_g < 0.0) {
      /* delta_g(outward) < 0, so inward transfer is penalised */
      k_b[(2 * i) + 1] *= exp(delta_g / (t * M_KB));
    } else if (delta_g > 0.0) {
      /* and vice versa */
      k_b[(2 * i)]     *= exp(-1.0 * delta_g / (t * M_KB));
    }
  }

  twa[1][0] = k_params[1]; /* k_trap */
  for (unsigned j = 2; j < side; j = j + n_s) {
    twa[1][j] = k_b[0]; /* RC -> LHC[0] */
    twa[j][1] = k_b[1]; /* RC <- LHC[0] */
    for (unsigned i = 0; i < n_s; i++) {
      if (i > 0) {
        twa[(i + j)][(i + j - 1)] = k_b[(2 * i) + 1];
      }
      if (i < (n_s - 1)) {
        twa[(i + j)][(i + j + 1)] = k_b[2 * (i + 1)];
      }
    }
  }

  /* now construct k */
  double* kd = calloc(side * side, sizeof(double));
  kd[0] -= k_params[2]; /* k_con */
  for (unsigned i = 0; i < side; i++) {
    if (i >= 2) {
      kd[(i * side) + i] -= k_params[0]; /* k_diss */
    }
    for (unsigned j = 0; j < side; j++) {
      if (i != j) {
        kd[(i * side) + j]  = twa[j][i];
        kd[(i * side) + i] -= twa[i][j];
      }
    }
  }

  /*
   * SATURATING RC CODE - NOT TESTED YET
   * new indexing convention:
   *
   * denote a config using occupancy numbers (n_s, n_RC, n_trap);
   * where n_s is actually a vector of occupancies for each subunit.
   * the set of possible transfers and rates depend on whether
   * the trap's open or closed, so we have double the possible
   * number of states from before: 2 * ((n_s * n_b) + 2).
   * collapse our occupancy numbers down into one index
   * in order to build a transfer matrix as follows:
   *
   * 0 -> vec(0) 0 0; 1 -> vec(0) 0 1; 
   * 2 -> vec(0) 1 0; 3 -> vec(0) 1 1;
   * then for i = 0, n_b * n_s:
   * (2i + 4) -> 1_i 0 0; (2i + 5) -> 1_i 0 1
   *
   * hopefully with comments this is parsable :)
   */

  double **new = calloc(2 * side, sizeof(double));
  for (unsigned i = 0; i < 2 * side; i++) {
    new[i] = calloc(2 * side, sizeof(double));
  }
  new[1][0] = k_params[2]; /* 0 0 1 -> 0 0 0 (k_con) */
  new[2][0] = k_params[0]; /* 0 1 0 -> 0 0 0 (k_diss) */
  new[2][1] = k_params[1]; /* 0 1 0 -> 0 0 1 (k_trap) */
  new[3][1] = k_params[0]; /* 0 1 1 -> 0 0 1 */
  new[3][2] = k_params[2]; /* 0 1 1 -> 0 1 0 */
  for (unsigned j = 4; j < 2 * side; j = j + 2 * n_s) {
    /* two pairs of RC <-> rates at the bottom of each branch */
    new[2][j]     = k_b[0]; /* 0 1 0   -> 1_i 0 0 */
    new[j][2]     = k_b[1]; /* 1_i 0 0 -> 0 1 0 */
    new[3][j + 1] = k_b[0]; /* 0 1 1   -> 1_i 0 1 */
    new[j + 1][3] = k_b[1]; /* 1_i 0 1 -> 0 1 1 */
    for (unsigned i = 0; i < n_s; i++) {
      unsigned ind = j + (2 * i);
      new[ind][0]       = k_params[0]; /* 1_i 0 0 -> 0 0 0 */
      new[ind + 1][1]   = k_params[0]; /* 1_i 0 1 -> 0 0 1 */
      new[ind + 1][ind] = k_params[2]; /* 1_i 0 1 -> 1_i 0 0 */
      if (i > 0) { /* pair of backward transfers down the branch */
        new[ind][ind - 2]     = k_b[(2 * i) + 1]; /* empty trap */
        new[ind + 1][ind - 1] = k_b[(2 * i) + 1]; /* full trap */
      }
      if (i < (n_s - 1)) { /* pair of forward transfers */
        new[ind][ind + 2]     = k_b[2 * (i + 1)]; /* empty */
        new[ind + 1][ind + 3] = k_b[2 * (i + 1)]; /* full */
      }
      new[0][ind]     = g[i]; /* 0 0 0 -> 1_i 0 0 */
      new[1][ind + 1] = g[i]; /* 0 0 1 -> 1_i 0 1 */
    }
  }

  double* newvec = calloc(pow((2 * side), 2), sizeof(double));
  for (unsigned i = 0; i < 2 * side; i++) {
    for (unsigned j = 0; j < 2 * side; j++) {
      if (i != j) {
        newvec[(i * (2 * side)) + j]  = new[j][i];
        newvec[(i * (2 * side)) + i] -= new[i][j];
      }
    }
  }

  gsl_matrix_view k = gsl_matrix_view_array(kd, side, side);
  gsl_linalg_LU_decomp(&k.matrix, perm, &signum);
  gsl_linalg_LU_solve(&k.matrix, perm, gamma, n_eq_gsl);
  for (unsigned i = 0; i < side; i++) {
    n_eq[i] = gsl_vector_get(n_eq_gsl, i);
  }

  if (isnan(n_eq[0])) {
    /* 
     * this shouldn't happen - if it does, something's gone wrong
     * earlier on somewhere, and it'll ruin the running averages in
     * the Python code. raise the SIGABRT to make it easier to debug
     */
    printf("NAN DETECTED\n");
    raise(SIGABRT);
  }
  nu_phi[0] = k_params[2] * n_eq[0];
  double sum_rate = 0.0;
  nu_phi[2] = 0.0; /* should already be. but just in case */
  for (unsigned i = 2; i < side; i++) {
    sum_rate += k_params[0] * n_eq[i];
    nu_phi[2] += n_eq[i];
  }
  nu_phi[1] = nu_phi[0] / (nu_phi[0] + sum_rate);

  for (unsigned i = 0; i < side; i++) {
    free(twa[i]);
    if (i < n_s + 1) {
      free(lines[i]);
    }
  }

  gsl_vector_free(gamma);
  gsl_vector_free(n_eq_gsl);
  gsl_permutation_free(perm);
  free(k_b);
  free(twa);
  free(g);
  free(lines);
  free(fp_y);
  free(kd);
}

int
main(int argc, char **argv)
{
  unsigned l_len = 4400;
  unsigned n_s = 5;
  unsigned n_b = 2;
  unsigned side = (n_b * n_s) + 2;
  FILE *fp = fopen("PHOENIX/Scaled_Spectrum_PHOENIX_2300K.dat", "r");
  char line[100];
  double *l = calloc(l_len, sizeof(double));
  double *ip_y = calloc(l_len, sizeof(double));
  for (unsigned i = 0; i < l_len; i++) {
    fgets(line, 100, fp); 
    sscanf(line, "%lf %le", &l[i], &ip_y[i]);
  }
  fclose(fp);
  double sigma = 5e-18;
  double k_params[5] = {1.0/4.0e-9, 1.0/5.0e-12,
    1.0/10.0e-3, 1.0/10.0e-12, 1.0/10.0e-12};
  double t = 300.0;
  unsigned *n_p = calloc(n_s + 1, sizeof(unsigned));
  double *lp = calloc(n_s + 1, sizeof(double));
  char *names[] = {"rc", "bchl_a", "chl_a", "chl_d", "r_pe", "chl_b"};
  double *n_eq = calloc(side, sizeof(double));
  double *nu_phi = calloc(3, sizeof(double));
  antenna(l, ip_y, sigma, k_params, t, n_p, lp, names, n_b, n_s, l_len,
            n_eq, nu_phi, 0);
  free(n_eq);
  free(nu_phi);
  free(n_p);
  free(lp);
  free(l);
  free(ip_y);
  return 0;
}

#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <stdio.h>

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

#if ! defined(M_HBAR)
	#define M_HBAR 1.054571817E-34L
#endif

#if ! defined(M_KB)
	#define M_KB 1.380649E-23L
#endif

/**********************************************************************
 
   Start of LU decomposition code

   i stole this from https://en.wikipedia.org/wiki/LU_decomposition
   because LAPACKE is annoying to install, GSL adds a load of complexity
   that we don't need, and doing it in fortran would make it a nightmare
   to interface with the Python (even more than it already is).

**********************************************************************/

/* 
 * INPUT: A - array of pointers to rows of a square matrix having dimension N
 * Tol - tolerance - detect failure when the matrix is near degenerate
 * OUTPUT: A contains a copy of both matrices L-E and U as A=(L-E)+U
 * such that P*A=L*U.
 * The permutation matrix is not stored as a matrix, but in an
 * integer vector P of size N+1 containing column indexes
 * where the permutation matrix has "1". The last element P[N]=S+N, where
 * S is the number of row exchanges needed for determinant computation,
 * det(P)=(-1)^S
 */
int LUPDecompose(double **A, int N, double Tol, int *P) {

    int i, j, k, imax;
    double maxA, *ptr, absA;

    for (i = 0; i <= N; i++)
        P[i] = i; //Unit permutation matrix, P[N] initialized with N

    for (i = 0; i < N; i++) {
        maxA = 0.0;
        imax = i;

        for (k = i; k < N; k++)
            if ((absA = fabs(A[k][i])) > maxA) {
                maxA = absA;
                imax = k;
            }

        if (maxA < Tol) return 0; //failure, matrix is degenerate

        if (imax != i) {
            //pivoting P
            j = P[i];
            P[i] = P[imax];
            P[imax] = j;

            //pivoting rows of A
            ptr = A[i];
            A[i] = A[imax];
            A[imax] = ptr;

            //counting pivots starting from N (for determinant)
            P[N]++;
        }

        for (j = i + 1; j < N; j++) {
            A[j][i] /= A[i][i];

            for (k = i + 1; k < N; k++)
                A[j][k] -= A[j][i] * A[i][k];
        }
    }

    return 1;  //decomposition done
}

/* INPUT: A,P filled in LUPDecompose; b - rhs vector; N - dimension
 * OUTPUT: x - solution vector of A*x=b
 */
void LUPSolve(double **A, int *P, double *b, int N, double *x) {

    for (int i = 0; i < N; i++) {
        x[i] = b[P[i]];

        for (int k = 0; k < i; k++)
            x[i] -= A[i][k] * x[k];
    }

    for (int i = N - 1; i >= 0; i--) {
        for (int k = i + 1; k < N; k++)
            x[i] -= A[i][k] * x[k];

        x[i] /= A[i][i];
    }
}

/* INPUT: A,P filled in LUPDecompose; N - dimension
 * OUTPUT: IA is the inverse of the initial matrix
 */
void LUPInvert(double **A, int *P, int N, double **IA) {

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            IA[i][j] = P[i] == j ? 1.0 : 0.0;

            for (int k = 0; k < i; k++)
                IA[i][j] -= A[i][k] * IA[k][j];
        }

        for (int i = N - 1; i >= 0; i--) {
            for (int k = i + 1; k < N; k++)
                IA[i][j] -= A[i][k] * IA[k][j];

            IA[i][j] /= A[i][i];
        }
    }
}

/* INPUT: A,P filled in LUPDecompose; N - dimension.
 * OUTPUT: Function returns the determinant of the initial matrix
 */
double LUPDeterminant(double **A, int *P, int N) {

    double det = A[0][0];

    for (int i = 1; i < N; i++)
        det *= A[i][i];

    return (P[N] - N) % 2 == 0 ? det : -det;
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

double*
absorption(double *l, double lambda_peak, double width, unsigned n)
{
  /* normalised gaussian absorption lineshape */
  double *work = calloc(n, sizeof(double));
  for (unsigned i = 0; i < n; i++) {
    work[i] = exp(-1.0 * pow(l[i] - lambda_peak, 2)/(2.0 * pow(width, 2)));
  }
  double norm = trapezoid(work, l, n);
  for (unsigned i = 0; i < n; i++) {
    work[i] /= norm;
  }
  return work;
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
            double t, unsigned *n_p, double *lp, double *width,
            unsigned n_b, unsigned n_s, unsigned l_len,
            double **twa, double* n_eq, double* nu_phi)
{
  /* 
   * length of n_p, lp and width should be n_s + 1; RC params are in [0].
   * k_params should be given as [k_diss, k_trap,  k_con, k_hop, k_lhc_rc].
   * l_len is the length of the l and ip_y arrays
   * twa should be a matrix with dims = (side, side)
   * and n_eq a vector with dim = (side).
   * twa should be initialised to zero.
   * nu_phi is double[2].
   */
  unsigned side = (n_b * n_s) + 2;
  double tol = 1e-6; /* LU decomposition tolerance for singularity */

  int*     perm  = calloc(side, sizeof(*perm));
  double*  k_b   = calloc(2 * n_s, sizeof(*k_b));
  double*  gamma = calloc(side, sizeof(*gamma));
  /* double*  n_eq  = calloc(side, sizeof(*n_eq)); */
  double*  g     = calloc(n_s, sizeof(*g));
  double** lines = calloc(n_s + 1, sizeof(*lines));
  /* double** twa   = calloc(side, sizeof(*twa)); */
  double** k     = calloc(side, sizeof(*k));
  for (unsigned i = 0; i < side; i++) {
    /* twa[i]   = calloc(side,  sizeof(*(twa[i]))); */
    k[i] = calloc(side,  sizeof(*(k[i])));
    if (i < n_s + 1) {
      lines[i] = calloc(l_len, sizeof(*(lines[i])));
    }
  }

  /* absorption rates */
  double *fp_y = calloc(l_len, sizeof(double));
  for (unsigned i = 0; i < l_len; i++) {
    fp_y[i] = ip_y[i] * l[i] * ((1.0E-9)/(M_H * M_C));
  }
  
  /* calculate lineshapes */
  for (unsigned i = 0; i < n_s + 1; i++) {
    lines[i] = absorption(l, lp[i], width[i], l_len);
    if (i > 0) { 
      /* 
       * add to the vector of photon inputs for later.
       * note that lines[0] is for the RC, so we skip that,
       * then subtract 1 to get back to correct indexing
       */
      g[i - 1] = n_p[i - 1] * sigma * overlap(l, fp_y, lines[i - 1], l_len);
      for (unsigned j = 2; j < side; j = j + n_s) {
        /* j is the set of start indices for each branch */
        gamma[i - 1 + j] = -1.0 * g[i - 1];
      }
    }
  }

  /* rate constants */
  /* THIS ALL NEEDS CHECKING - INDEXING, THE ORDERING OF DELTA_G */
  for (unsigned i = 0; i < n_s; i++) {
    double de = overlap(l, lines[i], lines[i + 1], l_len);
    double mean_w = (width[i] + width[i + 1]) / 2.0;
    de *= sqrt(4.0 * M_PI * mean_w);
    double n_ratio = n_p[i] / n_p[i + 1];
    /* next line calculates free energy change moving *outward* */
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
  k[0][0] -= k_params[2]; /* k_con */
  for (unsigned i = 0; i < side; i++) {
    if (i >= 2) {
      k[i][i] -= k[i][i] - k_params[0]; /* k_diss */
    }
    for (unsigned j = 0; j < side; j++) {
      k[i][j]  = twa[j][i];
      k[i][i] -= twa[i][j];
    }
  }

  LUPDecompose(k, side, tol, perm);
  LUPSolve(k, perm, gamma, side, n_eq);

  nu_phi[0] = k_params[2] * n_eq[0];
  double sum_rate = 0.0;
  for (unsigned i = 2; i < side; i++) {
    sum_rate += k_params[0] * n_eq[i];
  }
  nu_phi[1] = nu_phi[0] / (nu_phi[0] + sum_rate);

  for (unsigned i = 0; i < side; i++) {
    free(k[i]);
    if (i < n_s + 1) {
      free(lines[i]);
    }
  }
  free(k_b); free(gamma); free(g); free(lines);
  free(fp_y); free(k); free(perm);
}

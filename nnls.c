/*
 *
 * here's an (naive, from Lawson & Hanson, no fancy block pivoting shit)
 * implementation of non-negative least squares solver.
 * mainly doing this to try and understand it a bit better
 * 31/01/2024, callum gray
 *
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <signal.h>
#include <assert.h>
#include <time.h>

#define MAX_ITER 1000
#define SIGN(x) (x > 0) ? 1 : ((x < 0) ? -1 : 0)
#define RANDF() ((float)rand())/((float)(RAND_MAX))

void
assign_linear_slice(double *slice, double **A, int *slice_size,
    int xi, int xf, int yi, int yf)
{
  /* no array slicing in C :) */
  /* note the indexing here - need to make sure this is
   * consistent with the loops. currently it will assign
   * (xf - xi) * (yf - yi) elements as [xi, xf), [yi, yf) */
  (*slice_size) = 0;
  for (unsigned i = xi; i < xf; i++) {
    for (unsigned j = yi; j < yf; j++) {
      slice[(*slice_size)] = A[i][j];
      (*slice_size)++;
    }
  }
}

int
construct_householder(double *u, int ui, int n, double up)
{
  double cl = 0.0;
  for (unsigned i = 0; i < n; i++) {
    cl = (u[ui + i] > cl) ? u[ui + i] : cl;
  }
  if (cl < 0.0) {
    printf("what's happened here then. ui = %d, cl = %g\n", ui, cl);
    for (unsigned i = 0; i < n; i++) {
      printf("u[ui + %d] = %g\n", i, u[ui + i]);
    }
  }
  double cli = 1.0 / cl;
  double sm = 0.0;
  for (unsigned i = 0; i < n; i++) {
    sm += pow((u[ui + i] * cli), 2);
  }
  cl *= sqrt(sm);
  if (u[ui] > 0) cl *= -1.0;
  double res = u[ui] - cl;
  u[ui] = cl;
  return res; 
}

void
apply_householder(double *u, int ui,
    double *c, int ci, int n, double up)
{
  double b = up * u[0];
  if (b >= 0) return;
  b = 1.0 / b;
  double sm = c[0] * up;
  for (unsigned i = 1; i < n; i++) {
    sm += c[ci + i] * u[ui + i];
  }
  if (sm != 0.0) {
    sm *= b;
    c[0] += sm * up;
    for (unsigned i = 1; i < n; i++) {
      c[ci + i] += sm * u[ui + i];
    }
  }
}

void
orthogonal_rotmat(double a, double b, double *out)
{
  /* out should be double[3]: {c, s, sig} */
  double xr = 0.0;
  double yr = 0.0; 
  if (fabs(a) > fabs(b)) {
    xr = b / a;
    yr = sqrt(1 + pow(xr, 2));
    out[0] = (1.0 / yr) * (SIGN(a));
    out[1] = out[0] * xr;
    out[2] = fabs(a) * yr;
  } else if (b != 0) {
    xr = a / b;
    yr = sqrt(1 + pow(xr, 2));
    out[1] = (1.0 / yr) * (SIGN(b));
    out[0] = out[1] * xr;
    out[2] = fabs(b) * yr;
  } else {
    out[2] = 0.0;
    out[0] = 0.0;
    out[1] = 1.0;
  }
}

int
solve_triangular_system(double *zz, double *A, int n, int *idx, int nsetp, int j)
{
  for (unsigned i = 0; i < nsetp; i++) {
    int ip = nsetp - i;
    if (i != 0) {
      for (unsigned ii = 0; ii < ip; ii++) {
        zz[ii] -= A[(ii * n) + j] * zz[ip];
      }
    }
    j = idx[ip];
    zz[ip] /= A[(ip * n) + j];
  }
  return j;
}

void
largest_positive_dual(double *w, int *idx, int start, int stop, double *wmax, int *izmax)
{
  (*wmax) = 0.0;
  (*izmax) = 0;
  for (unsigned i = start; i < stop; i++) {
    unsigned j = idx[i];
    if (w[j] > (*wmax)) {
      (*wmax) = w[j];
      (*izmax) = i;
    }
  }
}

void
nnls(int m, int n, double *A, double *b, double *x,
    int *idx, double *rnorm, int mode, int *nsetp)
{
  /*
   * Solve NNLS problem min ||Ax - b|| subject to x_i > 0 for all i.
   * A -> double[m * n]. easier to just linear index it everywhere.
   * b, zz -> double[m]
   * x, w -> double[n]
   * idx -> int[n]
   * on output, rnorm will contain the residual norm,
   * nsetp will contain... something? not sure what its purpose is yet,
   * and x will contain the solution. everything else will be
   * destroyed, so don't count on using A or b afterwards.
   */
  double *w = calloc(n, sizeof(double));
  double *zz = calloc(m, sizeof(double));
  double *css = calloc(3, sizeof(double));
  double factor = 1e-2; /* check columns are sufficiently independent */
  mode = 1;
  int iter = 0;
  for (unsigned i = 0; i < n; i++) {
    x[i] = 0.0;
    idx[i] = i;
  }
  int iz2 = n;
  int iz1 = 0; /* this might not be correct - indexing. 1 in julia */
  int iz = 0;
  int j = 0;
  int nsp = 0;
  int up = 0;
  int terminated = 0;
  int izmax = 0;
  double wmax = 0.0;
  for (;;) {
    /* 
     * quit if all coefficients are in the solution,
     * or if m columns of A have been triangularised.
     */
    if (iz1 > iz2 || nsp >= m) {
      terminated = 1;
      break;
    }
    /* compute components of the negative gradient vector w */
    for (unsigned i = iz1; i < iz2; i++) {
      int idxi = idx[i];
      double sm = 0.0;
      for (unsigned l = (nsp + 1); l < m; l++) {
        sm += A[(l * n) + idxi] * b[l];
      }
      w[idxi] = sm;
    }

    for (;;) {
      /* find largest positive element of w */
      largest_positive_dual(w, idx, iz1, iz2, &wmax, &izmax);
      if (wmax <= 0.0) {
        terminated = 1;
        break;
      }
      iz = izmax;
      j = idx[iz];

      /* the sign of w[j] is ok for j to be moved to set p.
       * begin transformation and check diagonal element
       * to avoid near linear dependence */
      double Asave = A[(nsp * n) + j];
      /* get subset of A - I think this is correct but who knows */
      int xi = nsp * n + j;
      int size = m - nsp;
      up = construct_householder(A, xi, size, up);
      double unorm = 0.0;
      for (unsigned l = 0; l < nsp; l++) {
        unorm += pow(A[(l * n) + j], 2);
      }
      unorm = sqrt(unorm);
      double tt = (unorm + fabs(A[(nsp * n) + j] * factor) - unorm);
      if (tt > 0) {
        for (unsigned i = 0; i < m; i++) {
          zz[i] = b[i];
        }

        /* 
         * apply_householder slices the array from xi/xi2
         * so don't need to worry about malloc'ing slices
         * and then assigning them back
         */
        xi = nsp * n + j;
        size = m - nsp;
        int xi2 = nsp;
        apply_householder(A, xi, zz, xi2, size, up);
        double ztest = zz[nsp] / A[(nsp * n) + j];

        if (ztest > 0.0) {
          break;
        }
      }

      /* reject j as a candidate to be moved from z to p.
       * restore the element we modify in construct_householder,
       * set w[j] = 0.0; loop back to test dual coefficients again */
      A[(nsp * n) + j] = Asave;
      w[j] = 0.0;

    } /* inner for(;;) */

    if (terminated) {
      break;
    }

    /*
     * move index j from set z to set p. update b and indices,
     * apply householder transformations to columns in new z,
     * zero subdiagonal elements in column j, and set w[j] = 0.0.
     */
    idx[iz] = idx[iz1];
    idx[iz1] = j;
    iz1++;
    nsp++;

    if (iz1 <= iz2) {
      /* indexing on this loop will also be off i think */
      for (unsigned jz = iz1; jz < iz2; jz++) {
        int k = idx[jz];
        int xi = nsp * n + j;
        int size = m - nsp;
        int xi2 = nsp * n + k;
        apply_householder(A, xi, A, xi2, size, up);
      }
    }

    if (nsp != m) {
      for (unsigned l = nsp; l < m; l++) {
        A[(l * n) + j] = 0;
      }
    }
    w[j] = 0;

    /* solve triangular system. store solution in zz temporarily */
    j = solve_triangular_system(zz, A, n, idx, nsp, j);

    /* SECONDARY LOOP */
    for (;;) {
      iter++;
      if (iter > MAX_ITER) {
        mode = 3;
        terminated = 1;
        printf("NNLS quitting due to iteration count - nu_e -> 0\n");
        break;
      }

      /* 
       * check if all the new constrained coefficients are feasible.
       * if not, compute alpha
       */
      double alpha = 2.0;
      for (unsigned i = 0; i < nsp; i++) {
        int l = idx[i];
        if (zz[i] <= 0.0) {
          double t = -x[l] / (zz[i] - x[l]);
          if (alpha > t) {
            alpha = t;
            j = i;
          }
        }
      }
      /* 
       * if all new constrained coefficients are feasible,
       * alpha = 2.0 still from before the loop,
       * so exit back to main loop
       */
      if (alpha == 2.0) {
        break;
      }

      /*
       * otherwise, 0 < alpha < 1; interpolate between old x and new zz
       */
      for (unsigned i = 0; i < nsp; i++) {
        int l = idx[i];
        x[l] = x[l] + alpha * (zz[i] - x[l]);
      }

      /*
       * modify A and b and index arrays to move coefficient imove
       * from set p to set z
       */
      int imove = idx[j];

      for (;;) {
        x[imove] = 0;
        if (j != nsp) {
          j++;
          for (unsigned k = j; k < nsp; k++) {
            /* indexing again - julia's 1-based, and ranges are
             * inclusive at both ends, so this probably isn't
             * quite right? maybe one off at one end? */
            int ii = idx[k];
            idx[k - 1] = ii; /* again indexing */
            orthogonal_rotmat(A[(k - 1) * n + ii], A[(k * n) + ii], css);
            A[(k - 1) * n + ii] = css[2];
            A[(k * n) + ii] = 0.0;
            for (unsigned l = 0; l < n; l++) {
              if (l != ii) {
                /* apply procedure G2 (??) */
                int klm = (k - 1) * n + l;
                int kl = k * n + l;
                int temp = A[klm];
                A[klm] = css[0] * temp + css[1] * A[kl];
                A[kl] = -1.0 * css[1] * temp + css[0] * A[kl];
              }
            }
            int temp = b[k - 1];
            b[k - 1] = css[0] * temp + css[1] * b[k];
            b[k] = -1.0 * css[1] * temp + css[0] * b[k];
          }
        }
        nsp--;
        iz1--;
        idx[iz1] = imove;

        /* 
         * see if remaining cofficients in p are feasible.
         * they should be because of how alpha was determined,
         * so if any are infeasible it's due to roundoff error.
         * any that aren't positive we set to zero and move from p to z
         */
        int all_feasible = 1;
        for (unsigned ij = 0; ij < nsp; ij++) {
          j = ij;
          int ii = idx[j];
          if (x[ii] <= 0) {
            all_feasible = 0;
            break;
          }
        }
        if (all_feasible) break;

        printf("all feasible = %d\n", all_feasible);
      } /* for (;;) */
      for (unsigned i = 0; i < n; i++) {
        zz[i] = b[i];
      }
      j = solve_triangular_system(zz, A, n, idx, nsp, j);

    }

    if (terminated) break;
    /* end of secondary loop (??) */

    printf("nsp = %d\n", nsp);
    for (unsigned i = 0; i < nsp; i++) {
      x[idx[i]] = zz[i];
      printf("i, idx[i], x[idx[i]] = %d, %d, %g\n", i, idx[i], x[idx[i]]);
    }
    /* all new cofficients are positive - loop back */

  } /* outer for(;;) */

  /* end of main loop - termination */
  for (unsigned i = 0; i < n; i++) {
    printf("%g\n", x[i]);
  }

  /* compute residual norm */
  double sm = 0.0;
  if (nsp < m) {
    for (unsigned i = nsp; i < m; i++) {
      sm += pow(b[i], 2);
    }
  } else {
    for (unsigned i = 0; i < n; i++) {
      w[i] = 0.0;
    }
  }
  (*rnorm) = sqrt(sm);
  (*nsetp) = nsp;
  free(w);
  free(zz);
  free(css);
  return;
}

int
main(int argc, char** argv)
{

  unsigned i = 0;
  int n = 4;
  int m = 5;
  double *A = calloc(n * m, sizeof(double));
  double *b = calloc(m, sizeof(double));
  double *x = calloc(n, sizeof(double));
  double *res = calloc(n, sizeof(double));
  int *idx = calloc(n, sizeof(int));
  double rnorm = 0.0;
  int nsetp = 0;
  int mode = 0;
  srand(0);
  clock_t start = clock();
  int n_iter = atoi(argv[1]);

  while (i < n_iter) {
    printf("iteration %d: Ax = \n", i);
    for (unsigned j = 0; j < m; j++) {
      if (j < n) x[j] = RANDF();
      b[j] = 0.0;
    }
    for (unsigned j = 0; j < m; j++) {
      for (unsigned k = 0; k < n; k++) {
        A[(j * n) + k] = RANDF();
        b[j] += A[(j * n) + k] * x[k];
      }
    }
    for (unsigned j = 0; j < m; j++) {
      for (unsigned k = 0; k < n; k++) {
        printf("%6.4g ", A[(j * n) + k]);
      }
      if (j < n) {
        printf("\t%6.4g = %6.4g\n", x[j], b[j]);
      } else {
        printf("\t\t=%6.4g\n", b[j]);
      }
    }
    nnls(m, n, A, b, res, idx, &rnorm, mode, &nsetp);
    printf("Iteration %d finished\n", i);
    for (unsigned j = 0; j < m; j++) {
      for (unsigned k = 0; k < n; k++) {
        printf("%6.4g ", A[(j * n) + k]);
      }
        printf("\n");
    }
    double diff = 0.0;
    for (unsigned j = 0; j < n; j++) {
      printf("%6.4g \t %6.4g\n", x[j], res[j]);
      diff += fabs(x[j] - res[j]);
    }
    printf("iteration %d - sum of diff = %g\n", i, diff);
    i++;
  }

  clock_t diff = clock() - start;
  int msec = diff * 1000 / CLOCKS_PER_SEC;
  float msec_per_it = (float)msec / (float)argc;
  printf("Time for %d iterations = %dms; time per iteration = %g\n",
      n_iter, msec, msec_per_it);

  free(A);
  free(b);
  free(x);
  free(res);
  free(idx);
}

program main
  use iso_c_binding
  use nnls_solver
  implicit none
  integer(kind=c_int) :: i, j, k, n, m, mode, maxiter, nmax
  integer(kind=c_long) ::  start_time, end_time, count_rate
  real(kind=c_double) :: r, res, tol, diff, xdiff
  real(kind=c_double), dimension(:), allocatable :: b, x, x_ref
  real(kind=c_double), dimension(:, :), allocatable :: a
  character(len=100) :: outfile

  nmax = 0_c_int
  call random_init(.true., .true.)
  call system_clock(start_time, count_rate)
  do i = 1, 1000
    call random_number(r)
    ! at least 2 x 2 matrices
    n = 2_c_int + floor(20 * r) 
    call random_number(r)
    m = 2_c_int + floor(20 * r) 
    mode = 0_c_int
    res = 0.0_c_double
    maxiter = 3 * n
    tol = 1.0e-6_c_double

    allocate(b(m), source=0.0_c_double)
    allocate(x(n), source=0.0_c_double)
    allocate(x_ref(n))
    allocate(a(m, n))
    call random_number(a)
    call random_number(x_ref)
    b = matmul(a, x_ref)
    call nnls(a, b, x, mode, res, maxiter, tol)
    diff = norm2(matmul(A, x) - b)
    xdiff = norm2(x_ref - x)
    if ((mode.eq.-1_c_int).or.(diff.gt.1.0_c_double)) then
      write(outfile, '(a, i0.4, a)') "out/info_", i, ".txt"
      open(unit=20, file=outfile)
      nmax = nmax + 1
      write(20, *) "iteration ", i
      write(20, *) "x_ref = ", x_ref
      write(20, *) "x = ", x
      write(20, *) "shape(A) = ", shape(A)
      write(20, *) "A = "
      do j = 1, m
        write(20, *) (A(j, k), k = 1, n)
      end do
      write(20, *)
      write(20, *) "Ax = ", matmul(A, x)
      write(20, *) "b = ", b
      write(20, *) "diff = ", diff
      close(20)
    else
      write(*, '(a, i4, a, G10.3, a, G10.3)') "i = ", i,&
        " diff = ", diff, " xdiff = ", xdiff
    end if
    deallocate(b)
    deallocate(x)
    deallocate(x_ref)
    deallocate(a)
  end do
  call system_clock(end_time)
  write(*, *) "nmax = ", nmax
  write(*, '(a, f10.6)') "time elapsed = ",&
    real(end_time - start_time) / real(count_rate)

end program main

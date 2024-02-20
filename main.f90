program main
  use nnls_solver
  implicit none
  integer :: i, j, k, n, m, mode, maxiter, nmax
  real :: r, res, tol, diff
  real, dimension(:), allocatable :: b, x, x_ref
  real, dimension(:, :), allocatable :: a
  character(len=100) :: outfile

  nmax = 0
  call random_init(.true., .true.)
  do i = 1, 1000
    call random_number(r)
    ! at least 2 x 2 matrices
    n = 2 + floor(20 * r) 
    call random_number(r)
    m = 2 + floor(20 * r) 
    mode = 0
    res = 0.0
    maxiter = 3 * n
    tol = 1.0e-6

    allocate(b(m), source=0.0)
    allocate(x(n), source=0.0)
    allocate(x_ref(n))
    allocate(a(m, n))
    call random_number(a)
    call random_number(x_ref)
    b = matmul(a, x_ref)
    call nnls(a, b, x, mode, res, maxiter, tol)
    diff = sum(abs(matmul(A, x) - b))
    if ((mode.eq.-1).or.(diff.gt.1.0)) then
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
      write(*, '(a, i4, a, G10.3)') "i = ", i, " diff = ", diff
    end if
    deallocate(b)
    deallocate(x)
    deallocate(x_ref)
    deallocate(a)
  end do
  write(*, *) "nmax = ", nmax

end program main

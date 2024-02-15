program main
  use iso_c_binding
  use nnls_solver
  implicit none
  integer(kind=c_int) :: i, n, m, mode, maxiter
  real(kind=c_double) :: r, res, tol, diff
  real(kind=c_double), dimension(:), allocatable :: b, x, x_ref
  real(kind=c_double), dimension(:, :), allocatable :: a

  call random_init(.true., .true.)
  do i = 1, 1000
    call random_number(r)
    n = 1 + floor(20 * r) 
    call random_number(r)
    m = 1 + floor(20 * r) 
    mode = 0
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
    diff = sum(abs(x - x_ref))
    if (diff.gt.tol) then
      write(*, *) i, diff
    end if
    deallocate(b)
    deallocate(x)
    deallocate(x_ref)
    deallocate(a)
  end do

end program main

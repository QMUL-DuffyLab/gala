program main
  use nnls_solver
  implicit none
  integer :: i, n, m, mode, maxiter
  real :: r, res, tol, diff
  real, dimension(:), allocatable :: b, x, x_ref
  real, dimension(:, :), allocatable :: a

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
    diff = sum(abs(x - x_ref))
    write(*, *) i, diff
    deallocate(b)
    deallocate(x)
    deallocate(x_ref)
    deallocate(a)
  end do

end program main

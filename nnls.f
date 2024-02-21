! 13/02/2024
! @author: callum
! translated from scipy optimize nnls
module nnls_solver
  use iso_fortran_env
  use iso_c_binding
  implicit none
  public :: update_s, nnls

  contains

    subroutine update_s(s, ata, atb, p, n_true, s_p_min)&
        bind(C, name='update_s')
      implicit none
      real(kind=c_double), dimension(:, :) :: ata
      real(kind=c_double), dimension(:) :: atb, s
      logical(kind=c_bool), dimension(:) :: p
      integer(kind=c_int) :: n_true, i, info, lwork
      real(kind=c_double) :: s_p_min
      integer(kind=c_int), dimension(n_true) :: p_i, ipiv
      real(kind=c_double), dimension(n_true, n_true) :: atap
      real(kind=c_double), dimension(n_true) :: atbp
      real(kind=c_double), dimension(:), allocatable :: work

      if (size(ata, 1).ne.size(ata, 2)) then
        write (*,*) "ata not square"
        stop
      end if
      if (size(ata, 1).ne.size(p)) then
        write (*,*) "ata and p not same size"
        stop
      end if

      ! indices where p is true
      p_i = pack([(i, i = 1_c_int, size(p))], p)
      atap = ata(p_i, p_i)
      atbp = atb(p_i)

      allocate(work(1))
      call dsysv("U", n_true, 1, atap, n_true,&
                 ipiv, atbp, n_true, work, -1, info)
      lwork = int(work(1))
      deallocate(work)
      allocate(work(lwork))
      call dsysv("U", n_true, 1, atap, n_true,&
                 ipiv, atbp, n_true, work, lwork, info)
      ! atbp is now s[p]
      s_p_min = huge(0.0_c_double)
      do i = 1, n_true
        s(p_i(i)) = atbp(i)
        if (s(p_i(i)).lt.s_p_min) then
          s_p_min = s(p_i(i))
        end if
      end do
      deallocate(work)

    end subroutine update_s

    subroutine nnls(A, b, x, m, n, mode, res, maxiter, tol)&
        bind(C, name='nnls')
      implicit none
      integer(kind=c_int) :: m, n, iter, i, mode
      real(kind=c_double), dimension(m, n) :: A
      integer(kind=c_int) :: maxiter
      real(kind=c_double) :: tol

      logical(kind=c_bool), dimension(:), allocatable :: p
      real(kind=c_double), dimension(:, :), allocatable :: ata
      real(kind=c_double), dimension(:), allocatable :: atb, resid, s
      real(kind=c_double), dimension(m) :: b
      real(kind=c_double), dimension(n) :: x
      real(kind=c_double) :: s_p_min, alpha, alpha_min, res
      ! m = size(A, 1)
      ! n = size(A, 2)
      ! this will never happen now i guess
      if (size(b).ne.m) then
        write(*, *) size(A, 1), size(A, 2), size(b)
        write(*, *) "A and b dimensions incorrect"
        stop
      end if

      allocate(ata(n, n), source=0.0_c_double)
      ata = matmul(transpose(A), A)
      allocate(atb(n), source=0.0_c_double)
      allocate(resid(n), source=0.0_c_double)
      atb = matmul(b, A)
      resid = atb
      x = 0.0_c_double
      allocate(p(n))
      p = .false._c_bool
      allocate(s(n), source=0.0_c_double)

      mode = 1
      iter = 0
      do while ((.not.all(p)).and.&
        (any(merge(resid, 0.0_c_double,&
        (p.eqv..false._c_bool)).gt.tol)))

        where (p) resid = -huge(0.0_c_double)
        p(maxloc(resid, 1)) = .true._c_bool ! must specify dim to get scalar

        s = 0.0_c_double
        call update_s(s, ata, atb, p, count(p), s_p_min)

        do while ((iter.lt.maxiter).and.(s_p_min.le.tol))

          alpha_min = huge(0.0_c_double)
          do i = 1, n
            if ((p(i)).and.s(i).le.tol) then
              alpha = (x(i) / (x(i) - s(i))) 
              if (alpha.lt.alpha_min) then
                alpha_min = alpha
              end if
            end if
          end do
          x = x * (1.0_c_double - alpha_min)
          x = x + (alpha_min * s)
          where (x.lt.tol) p = .false._c_bool
          call update_s(s, ata, atb, p, count(p), s_p_min)
          where (.not.p) s = 0.0_c_double
          iter = iter + 1_c_int
          
        end do

        x = s
        resid = atb - matmul(ata, x)

      end do

      res = norm2(matmul(A, x) - b)
      deallocate(ata)
      deallocate(atb)
      deallocate(resid)
      deallocate(p)
      deallocate(s)
      if (all(x.eq.0.0_c_double)) mode = -1_c_int

    end subroutine nnls

end module nnls_solver

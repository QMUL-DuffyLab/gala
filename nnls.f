! 13/02/2024
! @author: callum
! translated from scipy optimize nnls
module nnls_solver
  use iso_fortran_env
  implicit none
  public :: update_s, nnls

  contains

    subroutine update_s(s, ata, atb, p, n_true, s_p_min)
      implicit none
      real, dimension(:, :) :: ata
      real, dimension(:) :: atb, s
      logical, dimension(:) :: p
      integer :: p_i_c, n_true, i, info, lwork
      real :: s_p_min
      integer, dimension(n_true) :: p_i, ipiv
      real, dimension(n_true, n_true) :: atap
      real, dimension(n_true) :: atbp
      real, dimension(:), allocatable :: work

      if (size(ata, 1).ne.size(ata, 2)) then
        write (*,*) "ata not square"
        stop
      end if
      if (size(ata, 1).ne.size(p)) then
        write (*,*) "ata and p not same size"
        stop
      end if

      p_i_c = 1
      do i = 1, size(p)
        if (p(i).eqv..true.) then
          p_i(p_i_c) = i
          p_i_c = p_i_c + 1
        end if
      end do

      atap = ata(p_i, p_i)
      atbp = atb(p_i)

      allocate(work(1))
      call ssysv("U", n_true, 1, atap, n_true,&
                 ipiv, atbp, n_true, work, -1, info)
      lwork = int(work(1))
      deallocate(work)
      allocate(work(lwork))
      call ssysv("U", n_true, 1, atap, n_true,&
                 ipiv, atbp, n_true, work, lwork, info)
      ! atbp is now s[p]
      s_p_min = huge(0.0)
      do i = 1, n_true
        s(p_i(i)) = atbp(i)
        if (s(p_i(i)).lt.s_p_min) then
          s_p_min = s(p_i(i))
        end if
      end do
      deallocate(work)

    end subroutine update_s

    subroutine nnls(A, b, x, mode, res, maxiter, tol)
      implicit none
      real, dimension(:, :) :: A
      integer :: maxiter
      real :: tol

      logical, dimension(:), allocatable :: p
      real, dimension(:, :), allocatable :: ata
      real, dimension(:), allocatable :: atb, resid, s
      real, dimension(size(A, 1)) :: b
      real, dimension(size(A, 2)) :: x
      integer :: m, n, iter, i, mode
      real :: s_p_min, alpha, alpha_min, res
      m = size(A, 1)
      n = size(A, 2)
      if (size(b).ne.m) then
        write(*, *) "A and b dimensions incorrect"
        stop
      end if

      allocate(ata(n, n), source=0.0)
      ata = matmul(transpose(A), A)
      allocate(atb(n), source=0.0)
      allocate(resid(n), source=0.0)
      atb = matmul(b, A)
      resid = atb
      x = 0.0
      allocate(p(n))
      p = .false.
      allocate(s(n), source=0.0)

      mode = 1
      iter = 0
      do while ((.not.all(p)).and.&
        (any(merge(resid, 0.0, (p.eqv..false.)).gt.tol)))

        where (p) resid = -huge(0.0)
        p(maxloc(resid, 1)) = .true. ! must specify dim to get scalar

        s = 0.0
        call update_s(s, ata, atb, p, count(p), s_p_min)

        do while ((iter.lt.maxiter).and.(s_p_min.le.tol))

          alpha_min = huge(0.0)
          do i = 1, n
            if ((p(i)).and.s(i).le.tol) then
              alpha = (x(i) / (x(i) - s(i))) 
              if (alpha.lt.alpha_min) then
                alpha_min = alpha
              end if
            end if
          end do
          x = x * (1.0 - alpha_min)
          x = x + (alpha_min * s)
          where (x.lt.tol) p = .false.
          call update_s(s, ata, atb, p, count(p), s_p_min)
          where (.not.p) s = 0.0
          iter = iter + 1
          
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
      if (all(x.eq.0.0)) mode = -1

    end subroutine nnls

end module nnls_solver

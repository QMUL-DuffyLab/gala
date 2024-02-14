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
      integer :: n_true, i, ldb, info, lwork
      real :: s_p_min
      integer, dimension(n_true) :: p_i, ipiv
      real, dimension(n_true, n_true) :: atap
      real, dimension(n_true) :: atbp
      real, dimension(n_true) :: sp
      real, dimension(:), allocatable :: work

      if (size(ata, 1).ne.size(ata, 2)) then
        write (*,*) "ata not square"
      end if
      if (size(ata, 1).ne.size(p)) then
        write (*,*) "ata and p not same size"
      end if
      do i = 1, size(ata, 1)
        if (p(i)) then
          p_i(i) = i
        end if
      end do

      atap = ata(p_i, p_i)
      atbp = atb(p_i)
      ! now solve
      ! scipy nnls assumes a symmetric matrix
      ! i guess this is true by construction? so dsysv
      allocate(work(1))
      call dsysv("U", n_true, 1, atap, n_true,&
                 ipiv, atbp, ldb, work, -1, info)
      lwork = int(work(1))
      deallocate(work)
      allocate(work(lwork))
      call dsysv("U", n_true, 1, atap, n_true,&
                 ipiv, atbp, ldb, work, lwork, info)
      ! atbp is now essentially s[p]
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
      integer, optional :: maxiter
      real, optional :: tol

      logical, dimension(:), allocatable :: p
      real, dimension(:, :), allocatable :: ata
      real, dimension(:), allocatable :: atb, resid, s, sp
      real, dimension(size(A, 1)) :: b
      real, dimension(size(A, 2)) :: x
      integer :: m, n, iter, k, n_true, i, mode
      real :: s_p_min, alpha, alpha_min, res
      m = size(A, 1)
      n = size(A, 2)
      if (size(b).ne.m) then
        write(*, *) "A and b dimensions incorrect"
      end if

      allocate(ata(n, n))
      ata = matmul(transpose(A), A)
      allocate(atb(n))
      allocate(resid(n))
      atb = matmul(b, A)
      resid = atb
      x = 0.0
      allocate(p(n))
      p = .false.
      allocate(s(n))
      s = 0.0

      if (.not.present(maxiter)) then
        maxiter = 3 * n
      end if
      if (.not.present(tol)) then
        tol = 10.0 * max(m, n) * tiny(0.0)
      end if

      iter = 0
      do while ((.not.all(p)).and.&
        (any(merge(resid, 0.0, (p.eqv..false.)) > tol)))

        where (p.eqv..true.) resid = -huge(0.0)
        k = maxloc(resid, 1) ! i think you have to specify dim to get a scalar
        p(k) = .true.

        s = 0.0
        n_true = count(p)
        call update_s(s, ata, atb, p, n_true, s_p_min)

        do while ((iter.lt.maxiter).and.(s_p_min.le.tol))
          alpha_min = huge(0.0)
          do i = 1, size(p)
            if ((p(i)).and.s(i).le.tol) then
              alpha = (x(i) / (x(i) - s(i))) 
              if (alpha.lt.alpha_min) then
                alpha_min = alpha
              end if
            end if
          end do
          x = x * (1.0 - alpha)
          x = x + alpha * s
          do i = 1, size(p)
            if (x(i).lt.tol) then
              p(i) = .false.
            end if
          end do
          n_true = count(p)
          call update_s(s, ata, atb, p, n_true, s_p_min)
          where (p.eqv..false.) s = 0
          iter = iter + 1
          
        end do

        x = s
        resid = atb - matmul(ata, x)

        if (iter.eq.maxiter) then
          ! do something
          x = 0.0
          res = 0.0
          mode = -1
        end if

      end do

      res = norm2(matmul(A, x) - b)
      deallocate(ata)
      deallocate(atb)
      deallocate(resid)
      deallocate(p)
      deallocate(s)

    end subroutine nnls

end module nnls_solver

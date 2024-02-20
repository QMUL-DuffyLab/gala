! 13/02/2024
! @author: callum
! translated from scipy optimize nnls
module nnls_solver
  use iso_fortran_env
  implicit none
  ! private :: update_s
  public :: update_s, nnls

  contains

    subroutine update_s(s, ata, atb, p, n_true, s_p_min)
      implicit none
      external :: dsysv
      real, dimension(:, :) :: ata
      real, dimension(:) :: atb, s
      logical, dimension(:) :: p
      integer :: p_i_c, n_true, i, ldb, info, lwork
      real :: s_p_min
      integer, dimension(n_true) :: p_i, ipiv
      real, dimension(n_true, n_true) :: atap
      real, dimension(n_true) :: atbp
      real, dimension(:), allocatable :: work

      if (size(ata, 1).ne.size(ata, 2)) then
        write (*,*) "ata not square"
      end if
      if (size(ata, 1).ne.size(p)) then
        write (*,*) "ata and p not same size"
      end if
      p_i_c = 1
      do i = 1, size(p)
        if (p(i).eqv..true.) then
          p_i(p_i_c) = i
          p_i_c = p_i_c + 1
        end if
      end do
      ldb = n_true

      atap = ata(p_i, p_i)
      atbp = atb(p_i)

      ! now solve
      ! scipy nnls assumes a symmetric matrix
      ! i guess this is true by construction? so dsysv
      allocate(work(1))
      ! write(*, *) "before first dsysv call", work, info, n_true
      call ssysv("U", n_true, 1, atap, n_true,&
                 ipiv, atbp, n_true, work, -1, info)
      lwork = int(work(1))
      deallocate(work)
      allocate(work(lwork))
      call ssysv("U", n_true, 1, atap, n_true,&
                 ipiv, atbp, ldb, work, lwork, info)
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

    ! subroutine antenna(n_p, lp, pigments, k_p, n_s, n_b)&
    !     bind(c, name="antenna")
    !   ! note - n_s should be n_s + 1 i.e. including rc
    !   implicit none
    !   integer :: n_s, n_b
    !   integer, dimension(n_s) :: n_p
    !   real, dimension(n_s) :: lp
    !   character, dimension(n_s) :: pigments
    !   real, dimension(5) :: k_p


    ! end subroutine antenna

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
      integer :: m, n, iter, k, n_true, i, mode
      real :: s_p_min, alpha, alpha_min, res, resid_max
      m = size(A, 1)
      n = size(A, 2)
      if (size(b).ne.m) then
        write(*, *) "A and b dimensions incorrect"
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
        (any(merge(resid, 0.0, (p.eqv..false.)) > tol)))

        do i = 1, n
          if (p(i).eqv..true.) then
            resid(i) = -huge(0.0)
          end if 
        end do
        resid_max = -1.0 * huge(0.0) + 1.0
        k = maxloc(resid, 1) ! you have to specify dim to get a scalar
        ! this line doesn't work properly. the code hangs because
        ! not all p's are true when they should be; one element of the
        ! array will still be false, but p(k) will return true. why?
        p(k) = .true.

        s = 0.0
        n_true = count(p)

        call update_s(s, ata, atb, p, n_true, s_p_min)

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
          x = x * (1.0 - alpha)
          x = x + alpha * s
          do i = 1, n
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

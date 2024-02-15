! 13/02/2024
! @author: callum
! translated from scipy optimize nnls
module nnls_solver
  use iso_fortran_env
  use iso_c_binding
  implicit none
  ! private :: update_s
  public :: update_s, nnls

  contains

    subroutine update_s(s, ata, atb, p, n_true, s_p_min)&
        bind(c, name="update_s")
      implicit none
      real(kind=c_double), dimension(:, :) :: ata
      real(kind=c_double), dimension(:) :: atb, s
      logical(kind=c_bool), dimension(:) :: p
      integer(kind=c_int) :: n_true, i, ldb, info, lwork
      real(kind=c_double) :: s_p_min
      integer(kind=c_int), dimension(n_true) :: p_i, ipiv
      real(kind=c_double), dimension(n_true, n_true) :: atap
      real(kind=c_double), dimension(n_true) :: atbp
      real(kind=c_double), dimension(:), allocatable :: work

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

    ! subroutine antenna(n_p, lp, pigments, k_p, n_s, n_b)&
    !     bind(c, name="antenna")
    !   ! note - n_s should be n_s + 1 i.e. including rc
    !   implicit none
    !   integer(kind=c_int) :: n_s, n_b
    !   integer(kind=c_int), dimension(n_s) :: n_p
    !   real(kind=c_double), dimension(n_s) :: lp
    !   character(kind=c_char), dimension(n_s) :: pigments
    !   real(kind=c_double), dimension(5) :: k_p


    ! end subroutine antenna

    subroutine nnls(A, b, x, mode, res, maxiter, tol) bind(c, name="nnls")
      implicit none
      real(kind=c_double), dimension(:, :) :: A
      integer(kind=c_int) :: maxiter
      real(kind=c_double) :: tol

      logical(kind=c_bool), dimension(:), allocatable :: p
      real(kind=c_double), dimension(:, :), allocatable :: ata
      real(kind=c_double), dimension(:), allocatable :: atb, resid, s
      real(kind=c_double), dimension(size(A, 1)) :: b
      real(kind=c_double), dimension(size(A, 2)) :: x
      integer(kind=c_int) :: m, n, iter, k, n_true, i, mode
      real(kind=c_double) :: s_p_min, alpha, alpha_min, res
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

      ! will have to do this in python - can't have
      ! optional variables in a c bound function
      ! if (.not.present(maxiter)) then
      !   maxiter = 3 * n
      ! end if
      ! if (.not.present(tol)) then
      !   tol = 10.0 * max(m, n) * tiny(0.0)
      ! end if

      iter = 0
      do while ((.not.all(p)).and.&
        (any(merge(resid, 0.0_c_double, (p.eqv..false.)) > tol)))

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

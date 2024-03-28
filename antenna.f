module antenna
  use json_module
  use nnls
  use iso_c_binding
  use iso_fortran_env, only: error_unit
  implicit none
  character(len=*), parameter :: pigment_file = "pigments/pigment_data.json"
  real(kind=CF), parameter :: h = 6.626e-34
  real(kind=CF), parameter :: c = 299792458.0
  real(kind=CF), parameter :: hcnm = (h * c) / (1.0e-9)
  real(kind=CF), parameter :: kb = 1.38e-23
  private
  public :: fitness_calc

  contains

    pure function integrate(x, y, n) result(r)
      integer(kind=CI), intent(in) :: n
      real(kind=CF), dimension(n), intent(in) :: x, y
      real(kind=CF) :: r
      r = sum((y(1 + 1:n - 0) + y(1 + 0:n - 1)) &
        * (x(1 + 1:n - 0) - x(1 + 0:n - 1)))/2
    end function

    subroutine lineshape(pigment, peak_offset, line, lp1, json, l, lsize)
      type(json_file) :: json
      integer(kind=CI), intent(in) :: lsize
      real(kind=CF), dimension(lsize), intent(in) :: l
      real(kind=CF), intent(in) :: peak_offset
      character(len=*), intent(in) :: pigment
      real(kind=CF), dimension(lsize), intent(out) :: line
      real(kind=CF), intent(out) :: lp1
      real(kind=CF), dimension(:), allocatable :: lp, width, amp
      integer(kind=CI) :: n_gauss, i
      real(kind=CF) :: r
      logical :: found
      call json%get(pigment // '.n_gauss', n_gauss, found)
      if (json%failed()) then
        call json%print_error_message(error_unit)
        stop
      end if
      ! if (.not.found) stop 1

      allocate(lp(n_gauss))
      allocate(width(n_gauss))
      allocate(amp(n_gauss))
      call json%get(pigment // '.lp', lp, found)
      if (json%failed()) then
        call json%print_error_message(error_unit)
        stop
      end if
      ! if (.not.found) stop 1
      call json%get(pigment // '.w', width, found)
      if (json%failed()) then
        call json%print_error_message(error_unit)
        stop
      end if
      ! if (.not.found) stop 1
      call json%get(pigment // '.amp', amp, found)
      if (json%failed()) then
        call json%print_error_message(error_unit)
        stop
      end if
      ! if (.not.found) stop 1

      lp = lp + peak_offset
      lp1 = lp(1)
      line = 0.0_CF
      do i = 1, n_gauss
        line = line + amp(i) * exp(-1.0_CF *&
          (l - lp(i))**2/(2.0_CF * width(i)**2))
      end do
      r = integrate(l, line, lsize)
      line = line / r
    end subroutine lineshape

    function overlap(line1, line2, l, lsize) result(r)
      integer(kind=CI), intent(in) :: lsize
      real(kind=CF), dimension(lsize), intent(in) :: line1, line2, l
      real(kind=CF) :: r
      r = integrate(l, line1 * line2, lsize)
    end function overlap

    pure function gibbs(l1, l2, n, T) result(r)
      real(kind=CF), intent(in) :: l1, l2, n, T
      real(kind=CF) :: h12, s12
      real(kind=CF) :: r
      h12 = hcnm * ((l1 - l2) / (l1 * l2))
      s12 = -kb * log(n)
      r = h12 - (s12 * T)
    end function gibbs
    
    subroutine fitness_calc(n_b, n_s, n_p, peak_offset, pigment,&
        k_params, temp, gamma_fac, l, ip_y, lsize, output)
    ! k_params = (k_diss, k_trap, k_con, k_hop, k_lhc_rc)
    integer(kind=CI), intent(in) :: n_b, n_s, lsize
    real(kind=CF), intent(in) :: temp
    integer(kind=CI), dimension(n_s + 1), intent(in) :: n_p
    real(kind=CF), dimension(n_s + 1), intent(in) :: peak_offset
    real(kind=CF), dimension(5), intent(in) :: k_params
    character(len=10), dimension(n_s + 1), intent(in) :: pigment
    real(kind=CF), dimension(lsize), intent(in) :: l, ip_y
    real(kind=CF), dimension(:, :), allocatable :: twa, k
    real(kind=CF), dimension(:), allocatable :: b, p_eq, n_eq
    real(kind=CF), dimension(n_s + 1, lsize) :: lines
    real(kind=CF), dimension(n_s + 1) :: lps
    real(kind=CF), dimension(n_s) :: g
    real(kind=CF), dimension(3) :: output
    real(kind=CF), dimension(2 * n_s) :: k_b
    real(kind=CF), dimension(lsize) :: fp_y
    type(json_file) :: json
    integer(kind=CI) :: i, j, ind, pen, mode, maxiter, side, io
    real(kind=CF) :: de, n, dg, res, tol, gamma_fac, sigma, rate, sgn,&
      nu_e_full, nu_e_low, phi_e_full, phi_e_low

    tol = 1.0e-13_CF
    sigma = 9e-20_CF
    rate = 0.0_CF
    fp_y = (ip_y * l) / hcnm

    call json%initialize()
    if (json%failed()) then
      call json%print_error_message(error_unit)
    end if

    call json%load(pigment_file)
    if (json%failed()) then
      call json%print_error_message(error_unit)
    end if

    ! lineshape and gamma calc
    do i = 1_CI, n_s + 1_CI
      call lineshape(trim(adjustl(pigment(i))), peak_offset(i),&
        lines(i, :), lps(i), json, l, lsize)
      if (i.gt.1_CI) then
        g(i - 1) = n_p(i) * sigma * overlap(fp_y, lines(i, :), l, lsize)
      end if
    end do

    ! overlap and dG calc, fill k_b
    do i = 1_CI, n_s
      de = overlap(lines(i, :), lines(i + 1, :), l, lsize)
      n = 1.0_CF * n_p(i) / n_p(i + 1)
      dg = gibbs(lps(i), lps(i + 1), n, temp)
      if (i.eq.1) then
        ! currently these two are identical but in principle not
        rate = k_params(5)
      else
        rate = k_params(4)
      end if
      rate = rate * de
      k_b(2 * i - 1) = rate
      k_b(2 * i) = rate
      ! penalise whichever rate corresponds to increasing free energy
      ! this covers all cases since if dg = 0.0 the exponential's 1
      pen = merge(2 * i - 1, 2 * i, dg > 0.0)
      sgn = merge(-1.0_CF, 1.0_CF, dg > 0.0)
      write(*, *) i, lps(i), de, rate, dg, pen, sgn
      k_b(pen) = k_b(pen) * exp(sgn * dg / (temp * kb))
    end do
    write(*, '(*(G10.3, 1X))') k_b

    ! assign transfer matrix twa and then k matrix from that
    side = (n_b * n_s) + 2_CI
    allocate(twa(2 * side, 2 * side), source=0.0_CF)
    allocate(k((2 * side) + 1, 2 * side), source=0.0_CF)
    ! careful of indexing here - 1-based vs 0-based
    twa(2, 1) = k_params(3) ! k_con
    twa(3, 1) = k_params(1) ! k_diss
    twa(3, 2) = k_params(2) ! k_trap
    twa(4, 2) = k_params(1)
    twa(4, 3) = k_params(3)
    ! outer loop - fill RC <-> branch rates
    do j = 5, 2 * side, 2 * n_s
      twa(3, j)     = k_b(1) ! 0 1 0   -> 1_i 0 0
      twa(j, 3)     = k_b(2) ! 1_i 0 0 -> 0 1 0
      twa(4, j + 1) = k_b(1) ! 0 1 1   -> 1_i 0 1
      twa(j + 1, 4) = k_b(2) ! 1_i 0 1 -> 0 1 1
      do i = 0, n_s - 1
        ! inner loop - decay and transfer rates along branch
        ind = j + (2 * i)
        twa(ind, 1)       = k_params(1) ! k_diss
        twa(ind + 1, 2)   = k_params(1)
        twa(ind + 1, ind) = k_params(3) ! k_con
        if (i.gt.0) then
          twa(ind, ind - 2)     = k_b(2 * (i + 1)) ! empty trap
          twa(ind + 1, ind - 1) = k_b(2 * (i + 1)) ! full trap
        end if
        if (i.lt.(n_s - 1)) then
          twa(ind, ind + 2)     = k_b(2 * (i + 1) + 1) ! empty
          twa(ind + 1, ind + 3) = k_b(2 * (i + 1) + 1) ! full
        end if
        twa(1, ind)     = g(i + 1) ! 0 0 0 -> 1_i 0 0
        twa(2, ind + 1) = g(i + 1) ! 0 0 1 -> 1_i 0 1
      end do
    end do

    ! assign k from twa
    do i = 1, 2 * side
      do j = 1, 2 * side
        if (i.ne.j) then
          k(i, j) = twa(j, i)
          k(i, i) = k(i, i) - twa(i, j)
        end if
      end do
      k(2 * side + 1, i) = 1.0
    end do
    open(newunit=io, file="out/kmat.dat")
    do i = 1, 2 * side + 1
      write(io, '(*(G10.3, 1X))') (k(i, j), j = 1, 2 * side)
    end do
    close(io)

    allocate(b((2 * side) + 1), source = 0.0_CF)
    allocate(p_eq(2 * side), source = 0.0_CF)
    b((2 * side) + 1) = 1.0_CF
    ! check mode!
    maxiter = 3 * (2 * side)
    call solve(k, b, p_eq, 2 * side + 1, 2 * side,&
      mode, res, maxiter, tol)
    if (mode.lt.0) then
      write(*, *) "nnls failed - high gamma"
      output = [0.0_CF, 0.0_CF, 0.0_CF]
      return
    end if
    ! write(*, '(*(G10.3, 1X))') p_eq

    allocate(n_eq(side), source = 0.0_CF)
    ! check indexing here - 1-based vs 0-based
    do i = 1, side
      n_eq(1) = n_eq(1) + p_eq(2 * i) ! P(1_i , 1)
      if (i.gt.1) then
        n_eq(i) = p_eq(2 * i - 1) + p_eq(2 * i) ! P(1_i, 0) + P(1_i, 1)
      end if
    end do
    nu_e_full  = k_params(3) * n_eq(1)
    phi_e_full = nu_e_full / (nu_e_full + (k_params(1) * sum(n_eq(2:))))

    ! now do the low gamma one
    g = gamma_fac * g / sum(g)
    do j = 5, 2 * side, 2 * n_s
      do i = 0, n_s - 1
        ind = j + (2 * i)
        twa(1, ind) = g(i + 1)
        twa(2, ind + 1) = g(i + 1)
      end do
    end do

    ! zero out k, otherwise the diagonal elements will be wrong
    k = 0.0

    ! check indexing
    do i = 1, 2 * side
      do j = 1, 2 * side
        if (i.ne.j) then
          k(i, j) = twa(j, i)
          k(i, i) = k(i, i) - twa(i, j)
        end if
      end do
      k(2 * side + 1, i) = 1.0
    end do
    b = 0.0_CF
    p_eq = 0.0_CF
    b((2 * side) + 1) = 1.0_CF
    call solve(k, b, p_eq, 2 * side + 1, 2 * side,&
      mode, res, maxiter, tol)
    if (mode.lt.0) then
      write(*, *) "nnls failed - low gamma"
      output = [0.0_CF, 0.0_CF, 0.0_CF]
      return
    end if

    n_eq = 0.0_CF
    ! check indexing
    do i = 1, side
      n_eq(1) = n_eq(1) + p_eq(2 * i) ! P(1_i , 1)
      if (i.gt.1) then
        n_eq(i) = p_eq(2 * i - 1) + p_eq(2 * i) ! P(1_i, 0) + P(1_i, 1)
      end if
    end do
    nu_e_low = k_params(3) * n_eq(1)
    phi_e_low = nu_e_low / (nu_e_low + (k_params(1) * sum(n_eq(2:))))

    output(1) = nu_e_full
    output(2) = phi_e_full
    output(3) = phi_e_low

    end subroutine fitness_calc
end module antenna

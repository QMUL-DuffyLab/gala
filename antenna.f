module antenna
  use json_module
  use nnls
  implicit none
  character(len=*), parameter :: pigment_file = "pigment/pigment_data.json"
  private
  public construct_k

  contains

    ! this won't work with bind(C) - allocatable arrays
    type pigment
      integer :: n_gauss
      real, allocatable, dimension(:) :: lp
      real, allocatable, dimension(:) :: width
      real, allocatable, dimension(:) :: amp
    end type pigment

    function new_pigment(n)
      integer, intent(in) :: n
      type(pigment) :: new_pigment
      new_pigment%n_gauss = n
      allocate(new_pigment%lp(n))
      allocate(new_pigment%width(n))
      allocate(new_pigment%amp(n))
    end function new_pigment

    function init_json(file) result(json)
      type(json_file), intent(out) :: json
      call json%initialize()
      call json%load(file)
    end function init_json

    pure function integrate(x, y, n) result(r)
      real, dimension(n), intent(in) :: x, y
      real :: r
      r = sum((y(1 + 1:n - 0) + y(1 + 0:n - 1)) &
        * (x(1 + 1:n - 0) - x(1 + 0:n - 1)))/2
    end function

    subroutine lineshape(pigment, peak_offset, line, lp1, json, l, lsize)
      type(json_file), intent(in) :: json
      real, dimension(lsize), intent(in) :: l
      real, intent(in) :: peak_offset
      character(len=*), intent(in) :: pigment
      real, dimension(lsize), intent(out) :: line
      real, intent(out) :: lp1
      real, dimension(:), allocatable :: lp, width, amp
      integer :: n_gauss, i
      logical :: found
      call json%get(pigment // '.n_gauss', n_gauss, found)
      if (.not.found) stop 1
      allocate(lp(n_gauss))
      allocate(width(n_gauss))
      allocate(amp(n_gauss))
      call json%get(pigment // '.lp', lp, found)
      if (.not.found) stop 1
      call json%get(pigment // '.width', width, found)
      if (.not.found) stop 1
      call json%get(pigment // '.amp', amp, found)
      if (.not.found) stop 1
      lp = lp - peak_offset
      lp1 = lp(1)
      line = 0.0
      do i = 1, n_gauss
        line = line + amp(i) * exp(-(l - lp(i))**2/(2.0 * width(i)**2))
      end do
    end subroutine lineshape

    function overlap(line1, line2, l, lsize) result(res)
      real, dimension(lsize), intent(in) :: line1, line2, l
    real, intent(out) :: res
      res = integrate(l, line1 * line2)
    end function overlap

    pure function gibbs(l1, l2, n, T) result(res)
      real, intent(in) :: l1, l2, T
      integer, intent(in) :: n
      real :: h12, s12
      real, intent(out) :: res
      h12 = hcnm * ((l1 - l2) / (l1 * l2))
      s12 = -kb * log(n)
      res = h12 - (s12 * T)
    end function gibbs
    
    subroutine fitness_calc(n_b, n_s, n_p, peak_offset, pigment,&
        k_params, temp, gamma_fac, l, ip_y, lsize, output)
    ! k_params = (k_diss, k_trap, k_con, k_hop, k_lhc_rc)
    integer, intent(in) :: n_b, n_s, lsize
    real, intent(in) :: temp
    integer, dimension(n_s), intent(in) :: n_p
    real, dimension(n_s), intent(in) :: peak_offset
    real, dimension(5), intent(in) :: k_params
    character, dimension(10, n_s), intent(in) :: pigment
    real, dimension(lsize), intent(in) :: l, ip_y
    real, dimension(:, :), allocatable :: twa, k
    real, dimension(:), allocatable :: b, p_eq
    real, dimension(n_s + 1, lsize) :: lines
    real, dimension(n_s + 1) :: lps
    real, dimension(n_s) :: g, gnorm
    real, dimension(3) :: output
    real, dimension(2 * n_s) :: k_b
    real, dimension(lsize) :: fp_y
    integer :: i, j, pen
    real :: de, n, dg, nu_e_full, nu_e_low, phi_e_full, phi_e_low

    fp_y = (ip_y * l) / hcnm

    do i = 1, n_s + 1
      ! lineshape and gamma calc
      call lineshape(trim(adjustl(pigment(i))), peak_offset(i),&
        lines(i), lps(i), json, l, lsize)
      if (i.gt.1) then
        g(i - 1) = n_p(i) * sigma * overlap(fp_y, lines(i), l, lsize)
      end if
    end do
    do i = 1, n_s
      ! overlap and dG calc, fill k_b
      de = overlap(lines(i), lines(i + 1), l, lsize)
      n = 1.0 * n_p(i) / n_p(i + 1)
      dg = gibbs(lps(i), lps(i + 1), n, temp)
      if (i.eq.1) then
        ! currently these two are identical but in principle not
        rate = k_params(5)
      else
        rate = k_params(4)
      end if
      rate = rate * de
      k_b(2 * i) = rate
      k_b((2 * i) + 1) = rate
      ! penalise whichever rate corresponds to increasing free energy
      pen = merge(2 * i, (2 * i) + 1, dg > 0.0)
      k_b(pen) = k_b(pen) * exp((-1.0 * sign(dg)) * dg / (T * kb))
    end do

    side = (n_b * n_s) + 2
    allocate(twa(2 * side, 2 * side))
    allocate(k((2 * side) + 1, 2 * side))
    ! careful of indexing here - 1-based vs 0-based
    twa(2, 1) = k_params(3) ! k_con
    twa(3, 1) = k_params(1) ! k_diss
    twa(3, 2) = k_params(2) ! k_trap
    twa(4, 2) = k_params(1)
    twa(4, 3) = k_params(3)
    do j = 5, 2 * side + 1, 2 * n_s
      ! outer loop - fill RC <-> branch rates
      twa(3, j)     = k_b(1) ! 0 1 0   -> 1_i 0 0
      twa(j, 3)     = k_b(2) ! 1_i 0 0 -> 0 1 0
      twa(4, j + 1) = k_b(1) ! 0 1 1   -> 1_i 0 1
      twa(j + 1, 4) = k_b(2) ! 1_i 0 1 -> 0 1 1
      do i = 1, n_s
        ! inner loop - decay and transfer rates along branch
        ind = j + (2 * i)
        twa(ind, 1)       = k_params(1) ! k_diss
        twa(ind + 1, 2)   = k_params(1)
        twa(ind + 1, ind) = k_params(3) ! k_con
        if (i.gt.1) then
          twa(ind, ind - 2)     = k_b((2 * i) + 1) ! empty trap
          twa(ind + 1, ind - 1) = k_b((2 * i) + 1) ! full trap
        end if
        if (i.lt.n_s) then
          twa(ind, ind + 2)     = k_b(2 * (i + 1)) ! empty
          twa(ind + 1, ind + 3) = k_b(2 * (i + 1)) ! full
        end if
        twa(1, ind)     = g(i) ! 0 0 0 -> 1_i 0 0
        twa(2, ind + 1) = g(i) ! 0 0 1 -> 1_i 0 1
      end do
    end do

    do i = 1, 2 * side
      do j = 1, 2 * side
        ! assign k from twa
        k(i, j) = twa(j, i)
        k(i, i) = k(i, i) - twa(i, j)
      end do
      k(2 * side + 1, i) = 1.0
    end do
    allocate(b((2 * side) + 1), source = 0.0)
    allocate(p_eq((2 * side) + 1), source = 0.0)
    b((2 * side) + 1) = 1.0
    ! check mode!
    p_eq = nnls(k, b)

    ! check indexing here - 1-based vs 0-based
    do i = 1, side
      n_eq(1) = n_eq(1) + p_eq((2 * i) + 1) ! P(1_i , 1)
      if (i.gt.1) then
        n_eq(i) = p_eq(2 * i) + p_eq((2 * i) + 1) ! P(1_i, 0) + P(1_i, 1)
      end if
    end do
    nu_e_full  = k_params(3) * n_eq(1)
    phi_e_full = nu_e / (nu_e + (k_params(1) * sum(n_eq(2:))))

    ! now do the low gamma one
    g = gamma_fac * g / sum(g)
    do j = 5, 2 * side + 1, 2 * n_s
      do i = 1, n_s
        ind = j + (2 * i)
        twa(1, ind) = g(i)
        twa(2, ind + 1) = g(i)
      end do
    end do

    ! zero out k, otherwise the diagonal elements will be wrong
    k = 0.0

    ! check indexing
    do i = 1, 2 * side
      do j = 1, 2 * side
        ! assign k from twa
        k(i, j) = twa(j, i)
        k(i, i) = k(i, i) - twa(i, j)
      end do
      k(2 * side + 1, i) = 1.0
    end do
    b = 0.0
    b((2 * side) + 1) = 1.0
    p_eq = nnls(k, b)

    n_eq = 0.0
    ! check indexing
    do i = 1, side
      n_eq(1) = n_eq(1) + p_eq((2 * i) + 1) ! P(1_i , 1)
      if (i.gt.1) then
        n_eq(i) = p_eq(2 * i) + p_eq((2 * i) + 1) ! P(1_i, 0) + P(1_i, 1)
      end if
    end do
    nu_e_low = k_params(3) * n_eq(1)
    phi_e_low = nu_e_low / (nu_e_low + (k_params(1) * sum(n_eq(2:))))

    output(1) = nu_e_high
    output(2) = phi_e_high
    output(3) = phi_e_low

    end subroutine fitness_calc
end module antenna

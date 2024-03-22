module antenna
  use json_module
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
    end function lineshape

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
    
    function construct_k(n_b, n_s, n_p, peak_offset, pigment, l, ip_y, lsize)&
        result(k)
    integer, intent(in) :: n_b, n_s, lsize
    integer, dimension(n_s), intent(in) :: n_p
    real, dimension(n_s), intent(in) :: peak_offset
    character, dimension(10, n_s), intent(in) :: pigment
    real, dimension(lsize), intent(in) :: l, ip_y
    real, dimension(:, :), allocatable :: twa, k
    real, dimension(n_s + 1, lsize) :: lines
    real, dimension(n_s + 1) :: lps
    real, dimension(n_s) :: g
    real, dimension(2 * n_s) :: k_b
    real, dimension(lsize) :: fp_y
    integer :: i, j, pen
    real :: de, n, dg

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
      ! need a separate peak function to get the actual peaks for this
      dg = gibbs(lps(i), lps(i + 1), n, T)
      ! and need k_params argument for this
      if (i.eq.1) then
        rate = k_params()
      else
        rate = k_params()
      end if
      rate = rate * de
      k_b(2 * i) = rate
      k_b((2 * i) + 1) = rate
      pen = merge(2 * i, (2 * i) + 1, dg > 0.0)
      k_b(pen) = k_b(pen) * exp((-1.0 * sign(dg)) * dg / (T * kb))
    end do

    side = (n_b * n_s) + 2
    allocate(twa(2 * side, 2 * side))
    allocate(k((2 * side) + 1, 2 * side))
    do j = 5, 2 * side + 1, 2 * n_s
      ! outer loop - fill RC - branch rates
      do i = 1, n_s
        ! inner loop - decay and transfer rates along branch
      end do
    end do

    do i = 1, 2 * side
      do j = 1, 2 * side
      ! assign k from twa
      end do
      k(2 * side + 1, i) = 1.0
    end do
    end function construct_k
end module antenna

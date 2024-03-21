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

    function lineshape(pigment, peak_offset, json, l, lsize)&
        result(line)
      type(json_file), intent(in) :: json
      real, dimension(lsize), intent(in) :: l
      real, intent(in) :: peak_offset
      character(len=*), intent(in) :: pigment
      real, dimension(lsize), intent(out) :: line
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

    pure function dG(l1, l2, n, T) result(res)
      real, intent(in) :: l1, l2, T
      integer, intent(in) :: n
      real :: h12, s12
      real, intent(out) :: res
      h12 = hcnm * ((l1 - l2) / (l1 * l2))
      s12 = -kb * log(n)
      res = h12 - (s12 * T)
    end function dG
    
    function construct_k(n_b, n_s, n_p, peak_offset, pigment, l, ip_y, lsize)&
        result(k)
    integer, intent(in) :: n_b, n_s, lsize
    integer, dimension(n_s), intent(in) :: n_p
    real, dimension(n_s), intent(in) :: peak_offset
    character, dimension(10, n_s), intent(in) :: pigment
    real, dimension(lsize), intent(in) :: l, ip_y
    real, dimension(:, :), allocatable :: twa, k
    integer :: i, j
    real, dimension(n_s + 1, lsize) :: lines(n_s + 1, lsize)
    real, dimension(n_s) :: g
    real, dimension(2 * n_s) :: k_b
    real, dimension(lsize) :: fp_y

    do i = 1, n_s + 1
      ! lineshape and gamma calc - careful of RC
    end do
    do i = 1, n_s
      ! overlap and dG calc, fill k_b
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

program main
  use antenna, only: fitness_calc
  use iso_fortran_env
  use iso_c_binding
  implicit none
  integer, parameter :: CF = c_double
  integer, parameter :: CI = c_int
  integer, parameter :: CB = c_bool

  integer(kind=CI) :: n_max, i, j, n_b, n_s, io, ios,&
    lsize, start_time, count_rate
  real(kind=CF), dimension(:), allocatable :: offset
  integer(kind=CI), dimension(:), allocatable :: n_p
  character(len=10), dimension(:), allocatable :: ps
  character(len=10), dimension(8) :: pigments
  real(kind=CF) :: r, temp, gamma_fac, k_params(5), output(3)
  real(kind=CF), dimension(:), allocatable :: l, ip_y

  write(*, *) "Starting"
  n_max = 1000_CI
  call random_init(.true., .true.)
  call system_clock(start_time, count_rate)
  write(*, *) "random_init done, clock started"

  open(newunit=io, file='spectra/PHOENIX/Scaled_Spectrum_PHOENIX_5800K.dat')
  ios = 0
  lsize = 0
  do while (ios.ne.iostat_end)
    read(io, *, iostat=ios)
    lsize = lsize + 1
  end do
  close(io)
  allocate(l(lsize), source=0.0_CF)
  allocate(ip_y(lsize), source=0.0_CF)
  open(newunit=io, file='spectra/PHOENIX/Scaled_Spectrum_PHOENIX_5800K.dat')
  do i = 1, lsize
    read(io, *, iostat=ios) l(i), ip_y(i)
  end do
  close(io)
  write(*, *) "spectrum file read in"

  pigments = [character(len=10) :: "chl_a", "chl_b", "chl_d",&
    "chl_f", "r_apc", "r_pc", "r_pe", "bchl_a"]
  k_params = [1.0_CF/4.0e-9_CF, 1.0_CF/5.0e-12_CF,&
    1.0_CF/1.0e-2_CF, 1.0_CF/1.0e-11_CF, 1.0_CF/1.0e-11_CF]
  gamma_fac = 1.0e-4_CF
  temp = 300.0_CF

  do i = 1, n_max
    write(*, *)
    write(*, *) "-------------------------"
    write(*, *)
    write(*, *) "i = ", i
    call random_number(r)
    n_b = 1 + floor(5.0 * r)
    call random_number(r)
    n_s = 1 + floor(5.0 * r)
    ! write(*, '(a, I3, 1X, a, I3)') "n_b = ", n_b, "n_s = ", n_s
    write(*,*) "n_b = ", n_b, "n_s = ", n_s
    allocate(offset(n_s + 1))
    allocate(n_p(n_s + 1))
    allocate(ps(n_s + 1))
    offset(1) = 0.0_CF
    n_p(1) = 10_CI
    ps(1) = 'rc'
    do j = 1, n_s
      call random_number(r)
      offset(j + 1) = -10.0_CF + 20.0_CF * r
      call random_number(r)
      n_p(j + 1) = 1_CI + floor(100_CI * r)
      call random_number(r)
      ps(j + 1) = pigments(1_CI + floor(8_CI * r))
    end do
    write(*, '(a, *(F8.3, 1X))') "offset = ", offset
    write(*, '(a, *(I3, 1X))') "n_p = ", n_p
    write(*, '(a, *(a, 1X))') "ps = ", ps
    call fitness_calc(n_b, n_s, n_p, offset, ps,&
        k_params, temp, gamma_fac, l, ip_y, lsize, output)
    write(*, '(a, *(F10.3, 1X))') "output = ", output
    deallocate(offset)
    deallocate(n_p)
    deallocate(ps)

  end do
  deallocate(l)
  deallocate(ip_y)

end program main

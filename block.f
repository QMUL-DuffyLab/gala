program block_hop
  use iso_fortran_env
  implicit none
  integer, parameter :: dp = REAL64
  logical :: allowed
  integer :: i, j, k, n, h, trial, nh_min, nh_max, n_trials
  real(dp) :: r, u, t, t_total, tmax, dt, k_pp, k_bb
  integer, dimension(6,4) :: neighbours
  character(100) :: filename
  integer, dimension(4) :: loc, new
  real, dimension(6) :: rates, cr
  real, dimension(:,:), allocatable :: avg, avg_sq

  filename = "out/averages_fortran.dat"
  dt = 1.0_dp
  tmax = 10000.0_dp
  n_trials = 10000
  nh_min = 2
  nh_max = 50
  k_pp = 1.0_dp / (1.0_dp * dt)
  k_bb = 1.0_dp / (10.0_dp * k_pp)
  rates = 0.0_dp
  cr = 0.0_dp

  allocate(avg(nh_max - nh_min + 1, nh_max - nh_min + 1))
  allocate(avg_sq(nh_max - nh_min + 1, nh_max - nh_min + 1))
  call random_init(.false., .true.)
  open(unit=19, file=trim(adjustl(filename)))

  do n = nh_min, nh_max
    ! do h = nh_min, nh_max
      i = n - nh_min + 1
      j = n - nh_min + 1
      t_total = 0.0_dp
      do trial = 1, n_trials
        loc(1) = 1
        call random_number(r)
        loc(2) = ceiling(r * n)
        call random_number(r)
        loc(3) = ceiling(r * n)
        call random_number(r)
        loc(4) = ceiling(r * n)
        t = 0.0_dp
        do while (loc(2).gt.1.and.t.lt.tmax)
          ! if loc(2) == 1 we're on the right face, so stop
          neighbours(1, :) = [loc(1), loc(2) - 1, loc(3), loc(4)]
          neighbours(2, :) = [loc(1), loc(2) + 1, loc(3), loc(4)]
          neighbours(3, :) = [loc(1), loc(2), loc(3) - 1, loc(4)]
          neighbours(4, :) = [loc(1), loc(2), loc(3) + 1, loc(4)]
          neighbours(5, :) = [loc(1), loc(2), loc(3), loc(4) - 1]
          neighbours(6, :) = [loc(1), loc(2), loc(3), loc(4) + 1]

          call random_number(r)
          k = ceiling(r * 6.0_dp)
          allowed = .true.

          if (neighbours(k, 2).eq.(n + 1)) then
            allowed = .false.
          end if
          if (neighbours(k, 3).eq.(n + 1)) then
            allowed = .false.
          end if
          if (neighbours(k, 3).eq.0) then
            allowed = .false.
          end if
          if (neighbours(k, 4).eq.(n + 1)) then
            allowed = .false.
          end if
          if (neighbours(k, 4).eq.0) then
            allowed = .false.
          end if

          if (allowed) then
            loc = neighbours(k, :)
          end if
          t = t + dt
        end do ! do while loc(1) = 1 and t < tmax
        t_total = t_total + t
      end do ! trials

      avg(i, j) = (t_total / n_trials)
      avg_sq(i, j) = (t_total / n_trials)**2
      write(*, *) "n = ", n, " avg = ", avg(i, j),&
        " avg^2 = ", avg_sq(i, j)
      write(19, '(I3.1, I3.1, E18.10, E18.10)') n, h, avg(i, j), avg_sq(i, j)
    ! end do
  end do
  close(19)

end program block_hop

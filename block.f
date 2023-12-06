program block_hop
  use iso_fortran_env
  implicit none
  integer, parameter :: dp = REAL64
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

  allocate(avg(nh_max - nh_min, nh_max - nh_min))
  allocate(avg_sq(nh_max - nh_min, nh_max - nh_min))
  call random_init(.false., .true.)
  open(unit=19, file=trim(adjustl(filename)))

  do n = nh_min, nh_max
    do h = nh_min, nh_max
      i = n - nh_min + 1
      j = h - nh_min + 1
      t_total = 0.0_dp
      do trial = 1, n_trials
        loc(1) = 1
        call random_number(r)
        loc(2) = ceiling(r * h)
        call random_number(r)
        loc(3) = ceiling(r * n)
        call random_number(r)
        loc(4) = ceiling(r * n)
        t = 0.0_dp
        do while (loc(1).eq.1.and.t.lt.tmax)
          neighbours(1, :) = [loc(1), loc(2) - 1, loc(3), loc(4)]
          neighbours(2, :) = [loc(1), loc(2) + 1, loc(3), loc(4)]
          neighbours(3, :) = [loc(1), loc(2), loc(3) - 1, loc(4)]
          neighbours(4, :) = [loc(1), loc(2), loc(3) + 1, loc(4)]
          neighbours(5, :) = [loc(1), loc(2), loc(3), loc(4) - 1]
          neighbours(6, :) = [loc(1), loc(2), loc(3), loc(4) + 1]
          do k = 1, 6
            rates(k) = k_pp
            if (neighbours(k, 2).eq.(h + 1)) then
              neighbours(k, 1) = 2
              neighbours(k, 2) = 1
              rates(k) = k_bb
            end if
            if (neighbours(k, 2).eq.0) then
              ! only one face is connected
              rates(k) = 0.0_dp
            end if
            if (neighbours(k, 3).eq.(n + 1)) then
              rates(k) = 0.0_dp
            end if
            if (neighbours(k, 3).eq.0) then
              rates(k) = 0.0_dp
            end if
            if (neighbours(k, 4).eq.(n + 1)) then
              rates(k) = 0.0_dp
            end if
            if (neighbours(k, 4).eq.0) then
              rates(k) = 0.0_dp
            end if
          end do

          do k = 1, 6
            cr(k) = sum(rates(1:k))
          end do
          call random_number(r)
          u = r * cr(6)
          k = 1
          do while (u.gt.cr(k))
            k = k + 1
          end do
          loc = neighbours(k, :)
          ! write(*, *) "rates = ", rates, "cr = ", cr
          call random_number(r)
          t = t + (1.0_dp / cr(6)) * log(1.0 / r)
          end do ! do while loc(1) = 1 and t < tmax
          t_total = t_total + t
      end do ! trials

      avg(i, j) = (t_total / n_trials)
      avg_sq(i, j) = (t_total / n_trials)**2
      write(*, *) "n, h = ", n, h, " avg = ", avg(i, j),&
        " avg^2 = ", avg_sq(i, j)
      write(19, '(I3.1, I3.1, F18.10, F18.10)') n, h, avg(i, j), avg_sq(i, j)
    end do
  end do
  close(19)

end program block_hop

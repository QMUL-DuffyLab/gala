using Luxor
using Colors

#=
to do:
- add n_p array
- convert RGB to RGBA with alpha ∝ n_p
- rescale the colourmap (we only have peaks between ~680 and ~550) and add colourbar
- add text to each block with a array of latex strings for each pigment
- make this into a script with n_b, n_s, λs, n_p as arguments
=#

n_b = 6
n_s = 4
λs = [650.0, 600.0, 550.0, 500.0]
θpad = (2π/360) * 5.0
rpad = 5.0
rcr = 40.0
dr = rcr * 1.1
cols = [RGB.(colormatch(λ)) for λ in λs]
@draw begin
    sethue("green")
    circle(Point(0.0, 0.0), rcr, action=:fillstroke)    
    for θ in range(0, step=2π/n_b, length=n_b)
        sep = (2π/(2n_b)) - (θpad / 2.0)
        max_angle =  (2π/360) * 45.0
        dθ = (sep > max_angle ? max_angle : sep)
        θ_i = θ - dθ
        θ_f = θ + dθ
        for i in range(0, step=1, length=n_s)
            r_i = (i * (dr + rpad))
            r_f = r_i + dr
            sethue(cols[i + 1])
             
            # easy way
            arc(polar(r_i, θ), dr, θ_i, θ_f)
            carc(polar(r_f, θ), dr, θ_f, θ_i)
            fillstroke()            
        end
    end
end

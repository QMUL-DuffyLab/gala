#=
08/03/24
author: @callum
=#

import Pkg
Pkg.add("Luxor")
Pkg.add("Colors")
Pkg.add("LaTeXStrings")
Pkg.add("MathTeXEngine")
Pkg.add("ArgParse")
Pkg.add("JSON")
import JSON
using LaTeXStrings
using Luxor
using Colors
using MathTeXEngine
using ArgParse

#=
to do: 
  - maybe think of a better way to display n_p?
  - ideally, move everything up and reduce the size - doesn't need the top and side borders
=#

function parse_cmds()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--n_b"
            help = "Number of branches n_b"
            arg_type = Int
            required = true
        "--n_s"
            help = "Number of subunits n_s"
            arg_type = Int
            required = true
        "--lambdas"
            help = "List of peak wavelengths λ"
            nargs = '+'
            required = true
        "--n_ps"
            help = "List of numbers of pigments n_p"
            nargs = '+'
            required = true
        "--names"
            help = "List of pigment names for each subunit"
            nargs = '+'
            required = true
        "--file"
            help = "Output filename"
            arg_type = String
            required = true
    end
    return parse_args(s)
end

function scale_λ(λ, scale = 2.0, min = 350.0, shift = 20.0)
    return scale .*(λ .- min) .+ shift
end

function major_tick(n, pos;
        startnumber = 0,
        finishnumber = 1,
        nticks = 1)
    ticklength = get_fontsize()
    line(pos, pos + polar(ticklength, π/2), :stroke)
    k = rescale(n, 0, nticks - 1, startnumber, finishnumber)
    ticklength = get_fontsize() * 1.35
    text("$(convert(Int, floor(k))) nm", pos + (0, ticklength), halign = :center, valign = :middle)
end

function colour_temp(n, pos;
        startnumber = 0,
        finishnumber = 1,
        nticks = 1,
        majorticklocations = [])
    k = rescale(n, 0, nticks - 1, startnumber, finishnumber)
    sethue(RGB.(colormatch(scale_λ(k))))
    box(pos, 20, 20, action = :fill)
end

function plot(n_b, n_s, λs, n_ps, names,
        name_positions="horizontal", output_file="test")
    # rescale λ - done for legibility but might have to change
    λs = scale_λ(λs)
    θpad = (2π/360) * 5.0
    rpad = 5.0
    rcr = 40.0
    dr = rcr * 1.1
    gamma = 2.2
    cols = [RGB.(colormatch(λ)) for λ in λs]
    rc_col = RGB.(colormatch(scale_λ(680.0)))
    size = round(2.5 * n_s) * (dr + rpad) + rcr + 100
    @svg begin
        fontsize(17)
        sethue(rc_col)
        circle(Point(0.0, 0.0), rcr, action=:fillstroke)
        name_positions == "horizontal" ? θ_start = 0 : θ_start = -π/2
        for θ in range(θ_start, step=2π/n_b, length=n_b)
            sep = (2π/(2n_b)) - (θpad / 2.0)
            # maximum block size - otherwise they look weird for small n_b
            max_angle =  (2π/360) * 45.0
            dθ = (sep > max_angle ? max_angle : sep)
            θ_i = θ - dθ
            θ_f = θ + dθ
            for i in range(0, step=1, length=n_s)
                r_i = (i * (dr + rpad))
                r_f = r_i + dr
                # get subunit colour
                setcolor(cols[i + 1])
                # create subunit block
                arc(polar(r_i, θ), dr, θ_i, θ_f)
                carc(polar(r_f, θ), dr, θ_f, θ_i)
                fillstroke()
            end
        end
        sethue("white")
        text("RC", O, halign = :center, valign = :middle)
        θ = θ_start
        for i in range(1, step=1, length=n_s)
            r = i * (dr + rpad) + dr/3.0
            name_positions == "horizontal" ? loc = Point(r, 5) : loc = Point(0, -r)
            # copied this from luxor docs - check luminance so that text is legible
            luminance = 0.2126 * cols[i].r^gamma + 0.7152 * cols[i].g^gamma + 0.0722 * cols[i].b^gamma
            (luminance > 0.5^gamma ? setcolor("black") : setcolor("white"))
            text(names[i], loc, halign=:center, valign=:bottom)
            θ_np = θ_start + (2π / n_b) + θpad / 2
            text(string(n_ps[i]), polar(r, θ_np), halign=:center, valign=:bottom, angle=θ_np - θpad)
        end
    sethue("black")
    colourbar_left = Point(-size/3, (n_s * (dr + rpad) + rcr + 20))
    colourbar_right = Point(size/3, (n_s * (dr + rpad) + rcr + 20))
    tickline(colourbar_left, colourbar_right,
            startnumber = 500, finishnumber = 700, 
            major = 3, minor = 30, 
            major_tick_function=major_tick,
            minor_tick_function=colour_temp)
    end size size output_file
end

function test()
    n_b = 5
    n_s = 5
    λs = [680.0, 660.0, 640.0, 620.0, 560.0]
    n_ps = [90, 60, 80, 20, 50]
    names = [L"$ \text{Chl}_{a} $", L"$ \text{Chl}_{b} $", L"$ \text{APC} $",
             L"$ \text{PC} $", L"$ \text{PE} $"]
    plot(n_b, n_s, λs, n_ps, names)
end

function main()
    args = parse_cmds()
    pigment_data = JSON.parsefile("pigments/pigment_data.json")
    n_ps = map(x -> parse(Int, x), args["n_ps"]) 
    λs = map(x -> parse(Float64, x), args["lambdas"]) 
    latex = [LaTeXString(pigment_data[n]["text"]) for n in args["names"]]
    plot(args["n_b"], args["n_s"], λs, n_ps, latex,
         "horizontal", args["file"])
end

main()

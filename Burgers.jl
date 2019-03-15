# Julia 1.0.1
# Solves the inviscid 1D Burgers' equation
#   ∂u/∂t + ∂f/∂x = 0
#   with periodic boundary conditions.
# By default, f = 1/2 * u^2.

import Plots
include("./Weno.jl")
using .Weno, BenchmarkTools

function burgers(u, cfl::Float64, t_max::Float64)
    # Courant condition
    dt::Float64 = dx / 2 * cfl
    t::Float64 = 0; counter::Int = 0
    op = zeros(nx)

    while t < t_max
        t += dt; counter += 1
        f = @. 1/2 * u^2
        λ = maximum(abs.(u))
        u .= Weno.runge_kutta(u, f, op, λ, dt)

        # Periodic boundary conditions
        u[end-0] = u[6]
        u[end-1] = u[5]
        u[end-2] = u[4]
        u[3] = u[end-3]
        u[2] = u[end-4]
        u[1] = u[end-5]
        # println("$counter   t = $t")
    end

    # plt = Plots.plot(x[cr], u[cr], linewidth=2, title="Burgers' Equation",
    #                  xaxis="x", yaxis="u(x)", label="u(x, $t)")
    # display(plt)

end

nx, dx, x, cr = Weno.grid(256, -5.0, 5.0, 3)
u = @. exp(-x^2)
# u = sech.(x)

@btime burgers(u, 0.3, 1.0);

# Julia 1.0.1
# Solves the inviscid 1D Burgers' equation
#   ∂u/∂t + ∂f/∂x = 0
#   with periodic boundary conditions.
# By default, f = 1/2 * u^2.

# import Plots
using BenchmarkTools
include("./WenoChar.jl")
using .WenoChar

mutable struct WenoParameter
    op::Vector{Float64}
    u1::Vector{Float64}
    u2::Vector{Float64}
    u3::Vector{Float64}
    fp::Vector{Float64}
    fm::Vector{Float64}
    ev::Float64
    ∂f::Float64
    fhat0::Float64
    fhat1::Float64
    fhat2::Float64
    w0::Float64
    w1::Float64
    w2::Float64
    IS0::Float64
    IS1::Float64
    IS2::Float64
    α0::Float64
    α1::Float64
    α2::Float64
    τ::Float64
    ϵ::Float64
end

function burgers(u, w, cfl::Float64, t_max::Float64)
    # Courant condition
    dt::Float64 = dx / 2 * cfl
    t::Float64 = 0; counter::Int = 0
    f = @. 1/2 * u^2

    while t < t_max
        t += dt; counter += 1
        w.ev = maximum(u)
        u .= WenoChar.runge_kutta(u, f, w, dt)

        # Periodic boundary conditions
        u[end-0] = u[6]
        u[end-1] = u[5]
        u[end-2] = u[4]
        u[3] = u[end-3]
        u[2] = u[end-4]
        u[1] = u[end-5]
        # println("$counter   t = $t")
        @. f = 1/2 * u^2
    end

    # plt = Plots.plot(x[cr], u[cr], linewidth=2, title="Burgers' Equation",
    #                  xaxis="x", yaxis="u(x)", label="u(x, $t)")
    # display(plt)
end

nx, dx, x, cr = WenoChar.grid(256, -5.0, 5.0, 3)
u = exp.(-x .^2)
# u = sech.(x)
w = WenoParameter(zeros(nx), zeros(nx), zeros(nx), zeros(nx), zeros(7), zeros(7),
                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1e-6)

@btime burgers(u, w, 0.3, 0.055);

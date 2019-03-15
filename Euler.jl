# Julia 1.0.1
# Solves the 1D Euler equations
#
#   ∂U/∂t + ∂F/∂x = 0
#
# where U = [ρ, ρu, E],
#       F = [ρu, ρu² + P, u(E + P)]
# and   E = P/(γ-1) + 1/2 * ρu².
# For a gas with 5 dofs, γ = 7/5.

import Plots
include("./Weno.jl")
using .Weno, BenchmarkTools

function primitive_to_conserved(ρ, u, P, γ)
    ρu = @. ρ * u # momentum
    E  = @. P / (γ-1.) + 1/2 * ρ * u^2 # energy
    return ρu, E
end

function conserved_to_primitive(ρ, ρu, E, γ)
    u = @. ρu / ρ
    P = @. (γ-1.) * (E - ρu^2 / (2 * ρ))
    return u, P
end

# Sod problem. x ∈ [-0.5, 0.5], t = 0.14
function sod()
    half::Int = nx ÷ 2
    for i = [:ρ, :u, :P]
        @eval $i = zeros(nx)
    end
    ρ[1:half] .= 1; ρ[half+1:end] .= 0.125
    P[1:half] .= 1; P[half+1:end] .= 0.1
    return ρ, u, P
end

# Lax problem. x ∈ [-0.5, 0.5], t = 0.13
function lax()
    half::Int = nx ÷ 2
    for i = [:ρ, :u, :P]
        @eval $i = zeros(nx)
    end
    ρ[1:half] .= 0.445; ρ[half+1:end] .= 0.5
    u[1:half] .= 0.698
    P[1:half] .= 3.528; P[half+1:end] .= 0.571
    return ρ, u, P
end

# Shu-Osher problem. x ∈ [-5.0, 5.0], t = 1.8
function shu_osher()
    half::Int = nx ÷ 10
    for i = [:ρ, :u, :P]
        @eval $i = zeros(nx)
    end
    @. ρ = 1.0 + 1/5 * sin(5x)
    ρ[1:half] .= 27/7
    u[1:half] .= 4*√(35)/9
    P[1:half] .= 31/3; P[half+1:end] .= 1.0
    return ρ, u, P
end

# Returns dt and λ = a+c
# CFL condition needs to be addressed
function timestep(ρ, u, P, cfl)
    a = maximum(abs.(u))
    c = maximum(.√(γ * P ./ ρ)) # sound speed
    λ = a + c
    return 0.1 * cfl * dx / λ, λ
end

function euler(ρ, u, P, γ::Float64, cfl::Float64, t_max::Float64)
    t::Float64 = 0; counter::Int = 0
    ρu, E = primitive_to_conserved(ρ, u, P, γ)
    op = zeros(nx)

    while t < t_max
        dt, λ = timestep(ρ, u, P, cfl)
        t += dt; counter += 1

        # Component-wise reconstruction
        # in the primitive variables
        f1 = ρu
        f2 = @. ρu^2 / ρ + P
        f3 = @. u * (E + P)
        ρ  .= Weno.runge_kutta(ρ,  f1, op, λ, dt)
        ρu .= Weno.runge_kutta(ρu, f2, op, λ, dt)
        E  .= Weno.runge_kutta(E,  f3, op, λ, dt)
        u, P = conserved_to_primitive(ρ, ρu, E, γ)

        println("$counter   t = $t")
    end

    plt = Plots.plot(x[cr], [ρ[cr], u[cr], P[cr]], linewidth=1, title="1D Euler Equations",
                     xaxis="x", label=["rho", "u", "P"])
    display(plt)
end

nx, dx, x, cr = Weno.grid(256, -0.5, 0.5, 3)
const γ = 7/5
ρ, u, P = sod()
# ρ, u, P = lax()
# ρ, u, P = shu_osher()
euler(ρ, u, P, γ, 0.6, 0.14)

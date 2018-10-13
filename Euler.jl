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
using .Weno

# Primitive variables
function basic_to_primitive(ρ, u, P)
    global γ
    ρu = ρ .* u # momentum
    E  = P ./ (γ-1.) + 1/2 * ρ .* u .^2 # energy
    return ρu, E
end

function primitive_to_basic(ρ, ρu, E)
    global γ
    u = ρu ./ ρ
    P = (γ-1.) * (E - ρu .^2 ./ (2 * ρ))
    return u, P
end

# Sod shock tube problem
# γ - ratio of specific heats
function sod_shock()
    global nx, γ
    half = nx ÷ 2
    
    for i = [:ρ, :u, :P]
        @eval $i = zeros(nx)
    end
    
    ρ[1:half] .= 1; ρ[half+1:end] .= 0.125
    P[1:half] .= 1; P[half+1:end] .= 0.1
    γ = 7/5

    return ρ, u, P, γ
end

# Returns dt and eval = a+c
# CFL condition needs to be addressed
function timestep(ρ, u, P, cfl)
    global dx
    a = maximum(abs.(u))
    c = maximum(.√(γ * P ./ ρ)) # sound speed
    eval = a + c
    return 0.1 * cfl * dx / eval, eval
end

function euler(ρ, u, P, γ::Float64, cfl::Float64, t_max::Float64)
    t::Float64 = 0; counter::Int = 0
    ρu, E = basic_to_primitive(ρ, u, P)

    while t < t_max
        dt, eval = timestep(ρ, u, P, cfl)
        t += dt; counter += 1

        # Component-wise reconstruction
        # in the primitive variables
        f1 = ρu
        f2 = ρu .^2 ./ ρ + P
        f3 = u .* (E + P)
        ρ  .= Weno.runge_kutta(ρ,  f1, eval, dt)
        ρu .= Weno.runge_kutta(ρu, f2, eval, dt)
        E  .= Weno.runge_kutta(E,  f3, eval, dt)
        u, P = primitive_to_basic(ρ, ρu, E)
        
        println("$counter   t = $t")
    end

    plt = Plots.plot(x[cr], [ρ[cr], u[cr], P[cr]], linewidth=1, title="Sod shock tube",
                     xaxis="x", label=["rho", "u", "P"])
    display(plt)
end

nx, dx, x, cr = Weno.grid(256, -0.5, 0.5, 3)
ρ, u, P, γ = sod_shock()
euler(ρ, u, P, γ, 0.6, 0.14)





    



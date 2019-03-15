# Solves the 1D Euler equations
#
#   ∂U/∂t + ∂F/∂x = 0
#
# where U = [ρ, ρu, E],
#       F = [ρu, ρu² + P, u(E + P)]
# and   E = P/(γ-1) + 1/2 * ρu².
# For a gas with 5 dofs, γ = 7/5.

include("./WenoChar.jl")
import .Weno
import Plots, Printf

function primitive_to_conserved(ρ, u, P, γ)
    ρu = @. ρ * u
    E  = @. P / (γ-1.) + 1/2 * ρ * u^2
    return ρu, E
end

function conserved_to_primitive(ρ, ρu, E, γ)
    u = @. ρu / ρ
    P = @. (γ-1) * (E - ρu^2 / (2 * ρ))
    return u, P
end

function h(γ, ρ, u, P)
    return γ/(γ-1) * P / ρ + 1/2 * u^2
end

function roe_average(ρ, u, h, γ)
    denom = √(ρ[i]) + √(ρ[i+1])
    u_avg = (√(ρ[i]) * u[i] + √(ρ[i+1]) * u[i+1]) / denom
    h_avg = (√(ρ[i]) * h[i] + √(ρ[i+1]) * h[i+1]) / denom
    c_avg = √((γ-1.) * (h_avg - 1/2 * u_avg^2))
    return u_avg, h_avg, c_avg
end

# Sod shock tube problem
function sod()
    half::Int = nx ÷ 2
    for i = [:ρ, :u, :P]
        @eval $i = zeros(nx)
    end
    ρ[1:half] .= 1; ρ[half+1:end] .= 0.125
    P[1:half] .= 1; P[half+1:end] .= 0.1
    return ρ, u, P
end

# Lax problem
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

# Shu-Osher problem. Set grid to x = [-5, 5].
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

# Returns dt and eval = a+c
# CFL condition needs to be addressed
function timestep(ρ, u, P, cfl)
    # global dx
    a = maximum(abs.(u))
    c = maximum(.√(γ * P ./ ρ)) # sound speed
    eval = a + c
    dt = 0.1 * cfl * dx / eval
    return dt, eval
end

function euler(ρ, u, P, γ, cfl, t_max)
    t = 0; counter = 0
    rk = preallocate_rungekutta_parameters()
    ρu, E = primitive_to_conserved(ρ, u, P, γ)
    # op = zeros(nx)

    while t < t_max
        dt, eval = timestep(ρ, u, P, cfl)
        t += dt; counter += 1

        # Component-wise reconstruction
        # in the primitive variables
        f1 = ρu
        f2 = @. ρu^2 / ρ + P
        f3 = @. u * (E + P)
        ρ  .= Weno.runge_kutta(ρ,  f1, op, eval, dt)
        ρu .= Weno.runge_kutta(ρu, f2, op, eval, dt)
        E  .= Weno.runge_kutta(E,  f3, op, eval, dt)
        u, P = conserved_to_primitive(ρ, ρu, E, γ)

        Printf.@printf("Iteration %d: t = %2.3f\n", counter, t)
    end

    plt = Plots.plot(x[cr], [ρ[cr], u[cr], P[cr]], title="1D Euler Equations",
                     xaxis="x", label=["rho", "u", "P"])
    display(plt)
end

function main(γ=7/5)
    gpar = Weno.grid(512, -0.5, 0.5, 3)

    # ρ, u, P = sod()
    # ρ, u, P = lax()
    # ρ, u, P = shu_osher()

    # euler(ρ, u, P, γ, 0.6, 0.13)
end

main()

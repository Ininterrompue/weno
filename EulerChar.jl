# Solves the 1D Euler equations
#
#   ∂U/∂t + ∂F/∂x = 0
#
# where U = [ρ, ρu, E],
#       F = [ρu, ρu² + P, u(E + P)]
# and   E = P/(γ-1) + 1/2 * ρu².
# For a gas with 5 dofs, γ = 7/5.

include("./WenoChar.jl")
import .WenoChar
import Printf
import Plots


struct Variables{T}
    ρ::Vector{T}    # density
    u::Vector{T}    # velocity
    P::Vector{T}    # pressure
    ρu::Vector{T}   # momentum
    E::Vector{T}    # energy
end

struct Fluxes{T}
    f1::Vector{T}   # real space
    f2::Vector{T}
    f3::Vector{T}
    g1::Vector{T}   # characteristic space
    g2::Vector{T}
    g3::Vector{T}
end


function preallocate_variables(grpar)
    for x in [:ρ, :u, :P, :ρu, :E]
        @eval $x = zeros($grpar.nx)
    end
    return Variables(ρ, u, P, ρu, E)
end

function preallocate_fluxes(grpar)
    for x in [:f1, :f2, :f3, :g1, :g2, :g3]
        @eval $x = zeros($grpar.nx)
    end
    return Fluxes(f1, f2, f3, g1, g2, g3)
end

# Sod shock tube problem
function sod!(U, grpar)
    half = grpar.nx ÷ 2
    U.ρ[1:half] .= 1; U.ρ[half+1:end] .= 0.125
    U.P[1:half] .= 1; U.P[half+1:end] .= 0.1
end

# Lax problem
function lax!(U, grpar)
    half = grpar.nx ÷ 2
    U.ρ[1:half] .= 0.445; U.ρ[half+1:end] .= 0.5
    U.u[1:half] .= 0.698
    U.P[1:half] .= 3.528; U.P[half+1:end] .= 0.571
end

# Shu-Osher problem. Set grid to x = [-5, 5].
function shu_osher!(U, grpar)
    tenth = grpar.nx ÷ 10
    @. U.ρ = 1.0 + 1/5 * sin(5 * grpar.x)
    U.ρ[1:tenth] .= 27/7
    U.u[1:tenth] .= 4/9 * sqrt(35)
    U.P[1:tenth] .= 31/3; U.P[tenth+1:end] .= 1.0
end

function primitive_to_conserved!(U, γ)
    @. U.ρu = U.ρ * U.u
    @. U.E  = U.P / (γ-1) + 1/2 * U.ρ * U.u^2
end

function conserved_to_primitive!(U, γ)
    @. U.u = U.ρu / U.ρ
    @. U.P = (γ-1) * (U.E - U.ρu^2 / 2ρ)
end

# function h(γ, ρ, u, P)
#     return γ/(γ-1) * P / ρ + 1/2 * u^2
# end

# function roe_average(ρ, u, h, γ)
#     denom = √(ρ[i]) + √(ρ[i+1])
#     u_avg = (√(ρ[i]) * u[i] + √(ρ[i+1]) * u[i+1]) / denom
#     h_avg = (√(ρ[i]) * h[i] + √(ρ[i+1]) * h[i+1]) / denom
#     c_avg = √((γ-1.) * (h_avg - 1/2 * u_avg^2))
#     return u_avg, h_avg, c_avg
# end

function max_eigval(U, γ)
    a = maximum(abs.(U.u))
    c = maximum(sqrt.(γ * U.P ./ U.ρ)) # sound speed
    return a + c
end

# CFL condition needs to be addressed
CFL_condition(eigval, cfl, grpar) = 0.1 * cfl * grpar.dx / eigval

function update_f_fluxes!(F, U)
    @. F.f1 = U.ρu
    @. F.f2 = U.ρu^2 / U.ρ + U.P
    @. F.f3 = U.u * (U.E + U.P)
end

function euler(γ=7/5, cfl=0.7, t_max=0.14)
    grpar = WenoChar.grid(128, -0.5, 0.5, 3)
    rkpar = WenoChar.preallocate_rungekutta_parameters(grpar)
    wepar = WenoChar.preallocate_weno_parameters(grpar)
    U = preallocate_variables(grpar)
    F = preallocate_fluxes(grpar)

    sod!(U, grpar)
    # lax!(U, grpar)
    # shu_osher!(U, grpar)

    primitive_to_conserved!(U, γ)
    t = 0.0; counter = 0

    while t < t_max
        wepar.ev = max_eigval(U, γ)
        dt = CFL_condition(wepar.ev, cfl, grpar)
        t += dt; counter += 1
        Printf.@printf("Iteration %d: t = %2.3f\n", counter, t)

        # Component-wise reconstruction in the primitive variables
        update_f_fluxes!(F, U)
        WenoChar.runge_kutta!(U.ρ,  F.f1, dt, grpar, rkpar, wepar)
        WenoChar.runge_kutta!(U.ρu, F.f2, dt, grpar, rkpar, wepar)
        WenoChar.runge_kutta!(U.E,  F.f3, dt, grpar, rkpar, wepar)
        conserved_to_primitive!(U, γ)
    end



    Printf.@printf("%d iterations. t_max = %2.3f.\n", counter, t)
    x = grpar.x; cr = grpar.cr
    plt = Plots.plot(x, [U.ρ, U.u, U.P], # [U.ρ[cr], U.u[cr], U.P[cr]],
                     title="1D Euler Equations",
                     xaxis="x", label=["rho", "u", "P"])
    display(plt)
end

euler()

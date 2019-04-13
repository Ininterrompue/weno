# Solves the 1D Euler equations
#
#   ∂U/∂t + ∂F/∂x = 0
#
# where U = [ρ, ρu, E],
#       F = [ρu, ρu² + P, u(E + P)]
# and   E = P/(γ-1) + 1/2 * ρu².
# For a gas with 5 dofs, γ = 7/5.

include("./Weno.jl")
import .Weno
import Printf, BenchmarkTools
import Plots


struct Variables{T}
    ρ::Vector{T}    # density
    u::Vector{T}    # velocity
    P::Vector{T}    # pressure
    ρu::Vector{T}   # momentum
    E::Vector{T}    # energy
end

struct AveragedVariables{T}
    ρ::Vector{T}
    u::Vector{T}
    P::Vector{T}
    cr::UnitRange{Int}
    J::Array{T, 3}
end

struct ProjectedVariables{T}
    ρ::Vector{T}
    ρu::Vector{T}
    E::Vector{T}
end

struct Fluxes{T}
    f1::Vector{T}
    f2::Vector{T}
    f3::Vector{T}
end

function preallocate_variables(grpar)
    for x in [:ρ, :u, :P, :ρu, :E, :ρ2, :ρu2, :E2]
        @eval $x = zeros($grpar.nx)
    end

    return Variables(ρ, u, P, ρu, E), ProjectedVariables(ρ2, ρu2, E2)
end

function preallocate_averaged_variables(grpar)
    cr = range(grpar.cr[1]-1, stop=grpar.cr[end])
    for x in [:ρ, :u, :P]
        @eval $x = zeros($grpar.nx+1)
    end
    J = zeros(3, 3, grpar.nx+1)
    return AveragedVariables(ρ, u, P, cr, J)
end

function preallocate_fluxes(grpar)
    for x in [:f1, :f2, :f3]
        @eval $x = zeros($grpar.nx)
    end
    return Fluxes(f1, f2, f3), Fluxes(f1, f2, f3)
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
    @. U.P = (γ-1) * (U.E - U.ρu^2 / 2U.ρ)
end

function arithmetic_average!(U, U_avg)
    for i in U_avg.cr
        U_avg.ρ[i] = 1/2 * (U.ρ[i] + U.ρ[i+1])
        U_avg.u[i] = 1/2 * (U.u[i] + U.u[i+1])
        U_avg.P[i] = 1/2 * (U.P[i] + U.P[i+1])
    end
end

sound_speed(P, ρ, γ) = sqrt(γ*P/ρ)
max_eigval(U, γ) = maximum(abs.(U.u) + sqrt.(γ * U.P ./ U.ρ))
CFL_condition(eigval, cfl, grpar) = 0.1 * cfl * grpar.dx / eigval

function update_fluxes!(F, U)
    @. F.f1 = U.ρu
    @. F.f2 = U.ρu^2 / U.ρ + U.P
    @. F.f3 = U.u * (U.E + U.P)
end

# J has dimensions (3, 3, nx+1).
# It is defined starting from the leftmost j-1/2.
function update_jacobian!(U, U_avg, U_proj, γ)
    arithmetic_average!(U, U_avg)

    for i in U_avg.cr

        U_avg.J[2, 1, i] = -(3-γ)/2 * U_avg.u[i]^2
        U_avg.J[3, 1, i] = (γ-2)/2 * U_avg.u[i]^3 - (U_avg.u[i] *
                           sound_speed(U_avg.P[i], U_avg.ρ[i], γ)^2 / (γ-1))
        U_avg.J[1, 2, i] = 1
        U_avg.J[2, 2, i] = (3-γ) * U_avg.u[i]
        U_avg.J[3, 2, i] = sound_speed(U_avg.P[i], U_avg.ρ[i], γ)^2 / (γ-1) +
                           (3-2γ)/2 * U_avg.u[i]^2
        U_avg.J[2, 3, i] = γ-1
        U_avg.J[3, 3, i] = γ * U_avg.u[i]
    end
end

function euler(γ=7/5, cfl=0.4, t_max=1.0)
    grpar = Weno.grid(1024, -5.0, 5.0, 3)
    rkpar = Weno.preallocate_rungekutta_parameters(grpar)
    wepar = Weno.preallocate_weno_parameters(grpar)
    U, V = preallocate_variables(grpar)
    U_avg = preallocate_averaged_variables(grpar)
    F, G = preallocate_fluxes(grpar)

    # sod!(U, grpar)
    # lax!(U, grpar)
    shu_osher!(U, grpar)

    primitive_to_conserved!(U, γ)
    t = 0.0; counter = 0

    while t < t_max
        wepar.ev = max_eigval(U, γ)
        dt = CFL_condition(wepar.ev, cfl, grpar)
        t += dt; counter += 1
        # Printf.@printf("Iteration %d: t = %2.3f\n", counter, t)

        # Component-wise reconstruction
        update_fluxes!(F, U)
        Weno.runge_kutta!(U.ρ,  F.f1, dt, grpar, rkpar, wepar)
        Weno.runge_kutta!(U.ρu, F.f2, dt, grpar, rkpar, wepar)
        Weno.runge_kutta!(U.E,  F.f3, dt, grpar, rkpar, wepar)
        conserved_to_primitive!(U, γ)

        # Characteristic-wise reconstruction
        # 1. Define the Jacobian in this file.
        # update_jacobian!(U, V, γ, grpar)
        # 2. Define the functions to convert to and from characteristic space
        #    in WenoChar.jl.
        # 3. Convert to char space in this file and call runge_kutta!()
        #    on V and G.
        # 4. Convert back to real space.
    end

    Printf.@printf("%d iterations. t_max = %2.3f.\n", counter, t)
    x = grpar.x; cr = grpar.cr
    plt = Plots.plot(x, [U.ρ, U.u, U.P],
                     title="1D Euler Equations",
                     xaxis="x", label=["rho", "u", "P"])
    display(plt)
end

# BenchmarkTools.@btime euler();
@time euler();

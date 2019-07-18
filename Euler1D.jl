# Solves the 1D Euler equations
#   ∂U/∂t + ∂F/∂x = 0
# where U = [ρ, ρu, E],
#       F = [ρu, ρu² + P, u(E + P)]
# and   E = P/(γ-1) + 1/2 * ρu².
# For a gas with 5 dofs, γ = 7/5.

include("./Weno.jl")
import .Weno
using Printf, LinearAlgebra
import Plots, BenchmarkTools


struct Variables{T}
    ρ::Vector{T}    # density
    u::Vector{T}    # velocity
    P::Vector{T}    # pressure
    ρu::Vector{T}   # momentum
    E::Vector{T}    # energy
end

mutable struct AveragedVariables{T}
    ρ::Vector{T}
    u::Vector{T}
    P::Vector{T}
    ρu::Vector{T}
    E::Vector{T}
    J::Matrix{T}
    eval::Vector{T}
    L::Matrix{T}
    R::Matrix{T}
end

struct ProjectedVariables{T}
    ρ::Vector{T}
    ρu::Vector{T}
    E::Vector{T}
end


function preallocate_variables(grpar)
    for x in [:ρ, :u, :P, :ρu, :E, :ρ2, :ρu2, :E2]
        @eval $x = zeros($grpar.nx)
    end

    return Variables(ρ, u, P, ρu, E), ProjectedVariables(ρ2, ρu2, E2)
end

function preallocate_averaged_variables(grpar)
    nx = grpar.nx
    for x in [:ρ, :u, :P, :ρu, :E]
        @eval $x = zeros($nx+1)
    end
    J = zeros(3, 3)
    eval = zeros(3)
    evecL = zeros(3, 3)
    evecR = zeros(3, 3)

    return AveragedVariables(ρ, u, P, ρu, E, J, eval, evecL, evecR)
end

function preallocate_fluxes(grpar)
    for x in [:f1, :f2, :f3, :g1, :g2, :g3]
        @eval $x = zeros($grpar.nx)
    end
    for x in [:h1, :h2, :h3, :j1, :j2, :j3]
        @eval $x = zeros($grpar.nx+1)
    end
    return ProjectedVariables(f1, f2, f3), ProjectedVariables(g1, g2, g3),
           ProjectedVariables(h1, h2, h3), ProjectedVariables(j1, j2, j3)
end

function preallocate_local()
    local_variables = ProjectedVariables(zeros(6), zeros(6), zeros(6))
    local_fluxes = ProjectedVariables(zeros(6), zeros(6), zeros(6))
    return local_variables, local_fluxes
end

"""
Sod shock tube problem
x = [-0.5, 0.5], t_max = 0.14
"""
function sod!(U, grpar)
    half = grpar.nx ÷ 2
    U.ρ[1:half] .= 1; U.ρ[half+1:end] .= 0.125
    U.P[1:half] .= 1; U.P[half+1:end] .= 0.1
end

"""
Lax problem
x = [-0.5, 0.5], t_max = 0.13
"""
function lax!(U, grpar)
    half = grpar.nx ÷ 2
    U.ρ[1:half] .= 0.445; U.ρ[half+1:end] .= 0.5
    U.u[1:half] .= 0.698
    U.P[1:half] .= 3.528; U.P[half+1:end] .= 0.571
end

"""
Shu-Osher problem
x = [-5, 5], t_max = 1.8
"""
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

function arithmetic_average!(U, U_avg, grpar)
    for i in grpar.cr_cell
        U_avg.ρ[i]  = 1/2 * (U.ρ[i]  + U.ρ[i+1])
        U_avg.u[i]  = 1/2 * (U.u[i]  + U.u[i+1])
        U_avg.P[i]  = 1/2 * (U.P[i]  + U.P[i+1])
        U_avg.ρu[i] = 1/2 * (U.ρu[i] + U.ρu[i+1])
        U_avg.E[i]  = 1/2 * (U.E[i]  + U.E[i+1])
    end
end

sound_speed(P, ρ, γ) = sqrt(γ*P/ρ)

max_eigval(U, γ) = @. $maximum(abs(U.u) + sound_speed(U.P, U.ρ, γ))

CFL_condition(eigval, cfl, grpar) = 0.1 * cfl * grpar.dx / eigval

function update_fluxes!(F, U)
    @. F.ρ = U.ρu
    @. F.ρu = U.ρu^2 / U.ρ + U.P
    @. F.E = U.u * (U.E + U.P)
end

"""
J has dimensions (3, 3, nx+1).
It is defined starting from the leftmost j-1/2.
"""
function update_jacobian!(i, U, U_avg, γ, grpar)
    arithmetic_average!(U, U_avg, grpar)

    U_avg.J[2, 1] = -(3-γ)/2 * U_avg.u[i]^2
    U_avg.J[3, 1] = (γ-2)/2 * U_avg.u[i]^3 - (U_avg.u[i] * (γ * U_avg.P[i] / U_avg.ρ[i]) / (γ-1))
    U_avg.J[1, 2] = 1
    U_avg.J[2, 2] = (3-γ) * U_avg.u[i]
    U_avg.J[3, 2] = (γ * U_avg.P[i] / U_avg.ρ[i]) / (γ-1) + (3-2γ)/2 * U_avg.u[i]^2
    U_avg.J[2, 3] = γ-1
    U_avg.J[3, 3] = γ * U_avg.u[i]
end

function update_local!(i, U, F, U_local, F_local)
    for j in 1:6
        U_local.ρ[j]  = U.ρ[i-3+j]
        U_local.ρu[j] = U.ρu[i-3+j]
        U_local.E[j]  = U.E[i-3+j]
        F_local.ρ[j]  = F.ρ[i-3+j]
        F_local.ρu[j] = F.ρu[i-3+j]
        F_local.E[j]  = F.E[i-3+j]
    end
end

function project_to_localspace!(i, U_avg, U, V, F, G)
    for j in i-2:i+3
        V.ρ[j]  = U_avg.L[1, 1] * U.ρ[j] + U_avg.L[1, 2] * U.ρu[j] + U_avg.L[1, 3] * U.E[j]
        V.ρu[j] = U_avg.L[2, 1] * U.ρ[j] + U_avg.L[2, 2] * U.ρu[j] + U_avg.L[2, 3] * U.E[j]
        V.E[j]  = U_avg.L[3, 1] * U.ρ[j] + U_avg.L[3, 2] * U.ρu[j] + U_avg.L[3, 3] * U.E[j]
        G.ρ[j]  = U_avg.L[1, 1] * F.ρ[j] + U_avg.L[1, 2] * F.ρu[j] + U_avg.L[1, 3] * F.E[j]
        G.ρu[j] = U_avg.L[2, 1] * F.ρ[j] + U_avg.L[2, 2] * F.ρu[j] + U_avg.L[2, 3] * F.E[j]
        G.E[j]  = U_avg.L[3, 1] * F.ρ[j] + U_avg.L[3, 2] * F.ρu[j] + U_avg.L[3, 3] * F.E[j]
    end
end

function project_to_realspace!(i, U, F, G)
    F.ρ[i]  = U.R[1, 1] * G.ρ[i] + U.R[1, 2] * G.ρu[i] + U.R[1, 3] * G.E[i]
    F.ρu[i] = U.R[2, 1] * G.ρ[i] + U.R[2, 2] * G.ρu[i] + U.R[2, 3] * G.E[i]
    F.E[i]  = U.R[3, 1] * G.ρ[i] + U.R[3, 2] * G.ρu[i] + U.R[3, 3] * G.E[i]
end

function plot_system(U, grpar, filename)
    x = grpar.x; cr = grpar.cr_mesh
    plt = Plots.plot(x, U.ρ, title="1D Euler equations", label="rho")
    Plots.plot!(x, U.u, label="u")
    Plots.plot!(x, U.P, label="P")
    display(plt)
    # Plots.png(plt, filename)
end

function euler(; γ=7/5, cfl=0.3, t_max=1.0)
    grpar = Weno.grid(size=1024, min=-5.0, max=5.0)
    rkpar = Weno.preallocate_rungekutta_parameters(grpar)
    wepar = Weno.preallocate_weno_parameters(grpar)
    U, V = preallocate_variables(grpar)
    U_avg = preallocate_averaged_variables(grpar)
    F, G, F_hat, G_hat = preallocate_fluxes(grpar)
    U_local, F_local = preallocate_local()

    # sod!(U, grpar)
    # lax!(U, grpar)
    shu_osher!(U, grpar)

    primitive_to_conserved!(U, γ)
    t = 0.0; counter = 0

    while t < t_max
        update_fluxes!(F, U)
        wepar.ev = max_eigval(U, γ)
        dt = CFL_condition(wepar.ev, cfl, grpar)
        t += dt 
        
        # Component-wise reconstruction
        # for i in grpar.cr_cell
        #     update_local!(i, U, F, U_local, F_local)
        #     F_hat.ρ[i]  = Weno.update_numerical_flux(U_local.ρ,  F_local.ρ,  wepar)
        #     F_hat.ρu[i] = Weno.update_numerical_flux(U_local.ρu, F_local.ρu, wepar)
        #     F_hat.E[i]  = Weno.update_numerical_flux(U_local.E,  F_local.E,  wepar)
        # end

        # Characteristic-wise reconstruction
        for i in grpar.cr_cell
            update_jacobian!(i, U, U_avg, γ, grpar)
            Weno.diagonalize_jacobian!(U_avg, grpar)
            project_to_localspace!(i, U_avg, U, V, F, G)
            update_local!(i, V, G, U_local, F_local)
            G_hat.ρ[i]  = Weno.update_numerical_flux(U_local.ρ,  F_local.ρ,  wepar)
            G_hat.ρu[i] = Weno.update_numerical_flux(U_local.ρu, F_local.ρu, wepar)
            G_hat.E[i]  = Weno.update_numerical_flux(U_local.E,  F_local.E,  wepar)
            project_to_realspace!(i, U_avg, F_hat, G_hat)
        end

        Weno.time_evolution!(U.ρ,  F_hat.ρ,  dt, grpar, rkpar) 
        Weno.time_evolution!(U.ρu, F_hat.ρu, dt, grpar, rkpar)
        Weno.time_evolution!(U.E,  F_hat.E,  dt, grpar, rkpar)

        conserved_to_primitive!(U, γ)

        counter += 1
        if counter % 100 == 0
            @printf("Iteration %d: t = %2.3f, dt = %2.3e\n", counter, t, dt)
        end
    end

    @printf("%d iterations. t_max = %2.3f.\n", counter, t)
    plot_system(U, grpar, "euler1d_shu_512")
end

# BenchmarkTools.@btime euler(t_max=0.14);
@time euler(t_max=1.8)

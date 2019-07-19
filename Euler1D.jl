# Solves the 1D Euler equations
#   ∂U/∂t + ∂F/∂x = 0
# where U = [ρ, ρu, E],
#       F = [ρu, ρu² + P, u(E + P)]
# and   E = P/(γ-1) + 1/2 * ρu².
# For a gas with 5 dofs, γ = 7/5.

include("./Weno.jl")
import .Weno
using Printf
import Plots, BenchmarkTools


struct Variables{T}
    ρ::Vector{T}    # density
    u::Vector{T}    # velocity
    P::Vector{T}    # pressure
    ρu::Vector{T}   # momentum
    E::Vector{T}    # energy
end

struct ConservedVariables{T}
    ρ::Vector{T}
    ρu::Vector{T}
    E::Vector{T}
end

struct StateVectors{T}
    Q::Variables{T}                  # real space
    Q_proj::ConservedVariables{T}    # characteristic space
    Q_local::ConservedVariables{T}   # local real space
end

struct Fluxes{T}
    F::ConservedVariables{T}         # physical flux, real space
    G::ConservedVariables{T}         # physical flux, characteristic space
    F_hat::ConservedVariables{T}     # numerical flux, real space
    G_hat::ConservedVariables{T}     # numerical flux, characteristic space
    F_local::ConservedVariables{T}   # local physical flux, real space
end

mutable struct FluxReconstruction{T}
    Q_avg::Variables{T}   # averaged quantities
    J::Matrix{T}          # Jacobian
    eval::Vector{T}       # eigenvalues
    L::Matrix{T}          # left eigenvectors
    R::Matrix{T}          # right eigenvectors
end


function preallocate_statevectors(grpar)
    for x in [:ρ, :u, :P, :ρu, :E, :ρ2, :ρu2, :E2]
        @eval $x = zeros($grpar.nx)
    end
    for x in [:ρ3, :ρu3, :E3]
        @eval $x = zeros(6)
    end
    Q = Variables(ρ, u, P, ρu, E)
    Q_proj = ConservedVariables(ρ2, ρu2, E2)
    Q_local = ConservedVariables(ρ3, ρu3, E3)
    return StateVectors(Q, Q_proj, Q_local)
end

function preallocate_fluxes(grpar)
    for x in [:f1, :f2, :f3, :g1, :g2, :g3]
        @eval $x = zeros($grpar.nx)
    end
    for x in [:f1_hat, :f2_hat, :f3_hat, :g1_hat, :g2_hat, :g3_hat]
        @eval $x = zeros($grpar.nx+1)
    end
    for x in [:f1_local, :f2_local, :f3_local]
        @eval $x = zeros(6)
    end
    F = ConservedVariables(f1, f2, f3)
    G = ConservedVariables(g1, g2, g3)
    F_hat = ConservedVariables(f1_hat, f2_hat, f3_hat)
    G_hat = ConservedVariables(g1_hat, g2_hat, g3_hat)
    F_local = ConservedVariables(f1_local, f2_local, f3_local)
    return Fluxes(F, G, F_hat, G_hat, F_local)
end

function preallocate_fluxreconstruction(grpar)
    nx = grpar.nx
    for x in [:ρ, :u, :P, :ρu, :E]
        @eval $x = zeros($nx+1)
    end
    Q_avg = Variables(ρ, u, P, ρu, E)
    J = zeros(3, 3)
    eval = zeros(3)
    evecL = zeros(3, 3)
    evecR = zeros(3, 3)
    return FluxReconstruction(Q_avg, J, eval, evecL, evecR)
end

"""
Sod shock tube problem
x = [-0.5, 0.5], t_max = 0.14
"""
function sod!(Q, grpar)
    half = grpar.nx ÷ 2
    Q.ρ[1:half] .= 1; Q.ρ[half+1:end] .= 0.125
    Q.P[1:half] .= 1; Q.P[half+1:end] .= 0.1
end

"""
Lax problem
x = [-0.5, 0.5], t_max = 0.13
"""
function lax!(Q, grpar)
    half = grpar.nx ÷ 2
    Q.ρ[1:half] .= 0.445; Q.ρ[half+1:end] .= 0.5
    Q.u[1:half] .= 0.698
    Q.P[1:half] .= 3.528; Q.P[half+1:end] .= 0.571
end

"""
Shu-Osher problem
x = [-5, 5], t_max = 1.8
"""
function shu_osher!(Q, grpar)
    tenth = grpar.nx ÷ 10
    @. Q.ρ = 1.0 + 1/5 * sin(5 * grpar.x)
    Q.ρ[1:tenth] .= 27/7
    Q.u[1:tenth] .= 4/9 * sqrt(35)
    Q.P[1:tenth] .= 31/3; Q.P[tenth+1:end] .= 1.0
end

function primitive_to_conserved!(Q, γ)
    @. Q.ρu = Q.ρ * Q.u
    @. Q.E  = Q.P / (γ-1) + 1/2 * Q.ρ * Q.u^2
end

function conserved_to_primitive!(Q, γ)
    @. Q.u = Q.ρu / Q.ρ
    @. Q.P = (γ-1) * (Q.E - Q.ρu^2 / 2Q.ρ)
end

function arithmetic_average!(Q, Q_avg, grpar)
    for i in grpar.cr_cell
        Q_avg.ρ[i]  = 1/2 * (Q.ρ[i]  + Q.ρ[i+1])
        Q_avg.u[i]  = 1/2 * (Q.u[i]  + Q.u[i+1])
        Q_avg.P[i]  = 1/2 * (Q.P[i]  + Q.P[i+1])
        Q_avg.ρu[i] = 1/2 * (Q.ρu[i] + Q.ρu[i+1])
        Q_avg.E[i]  = 1/2 * (Q.E[i]  + Q.E[i+1])
    end
end

sound_speed(P, ρ, γ) = sqrt(γ*P/ρ)

max_eigval(Q, γ) = @. $maximum(abs(Q.u) + sound_speed(Q.P, Q.ρ, γ))

CFL_condition(eigval, cfl, grpar) = 0.1 * cfl * grpar.dx / eigval

function update_physical_fluxes!(F, Q)
    @. F.ρ  = Q.ρu
    @. F.ρu = Q.ρ * Q.u^2 + Q.P
    @. F.E  = Q.u * (Q.E + Q.P)
end

"""
J has dimensions (3, 3, nx+1).
It is defined starting from the leftmost j-1/2.
"""
function update_jacobian!(i, Q, flxrec, γ, grpar)
    Q_avg = flxrec.Q_avg; J = flxrec.J

    arithmetic_average!(Q, Q_avg, grpar)
    J[2, 1] = -(3-γ)/2 * Q_avg.u[i]^2
    J[3, 1] = (γ-2)/2 * Q_avg.u[i]^3 - (Q_avg.u[i] * (γ * Q_avg.P[i] / Q_avg.ρ[i]) / (γ-1))
    J[1, 2] = 1
    J[2, 2] = (3-γ) * Q_avg.u[i]
    J[3, 2] = (γ * Q_avg.P[i] / Q_avg.ρ[i]) / (γ-1) + (3-2γ)/2 * Q_avg.u[i]^2
    J[2, 3] = γ-1
    J[3, 3] = γ * Q_avg.u[i]
end

function update_local!(i, Q, F, Q_local, F_local)
    for j in 1:6
        Q_local.ρ[j]  = Q.ρ[i-3+j]
        Q_local.ρu[j] = Q.ρu[i-3+j]
        Q_local.E[j]  = Q.E[i-3+j]
        F_local.ρ[j]  = F.ρ[i-3+j]
        F_local.ρu[j] = F.ρu[i-3+j]
        F_local.E[j]  = F.E[i-3+j]
    end
end

function project_to_localspace!(i, state, flux, flxrec)
    Q = state.Q; Q_proj = state.Q_proj
    F = flux.F; G = flux.G
    L = flxrec.L 
    for j in i-2:i+3
        Q_proj.ρ[j]  = L[1, 1] * Q.ρ[j] + L[1, 2] * Q.ρu[j] + L[1, 3] * Q.E[j]
        Q_proj.ρu[j] = L[2, 1] * Q.ρ[j] + L[2, 2] * Q.ρu[j] + L[2, 3] * Q.E[j]
        Q_proj.E[j]  = L[3, 1] * Q.ρ[j] + L[3, 2] * Q.ρu[j] + L[3, 3] * Q.E[j]
        G.ρ[j]  = L[1, 1] * F.ρ[j] + L[1, 2] * F.ρu[j] + L[1, 3] * F.E[j]
        G.ρu[j] = L[2, 1] * F.ρ[j] + L[2, 2] * F.ρu[j] + L[2, 3] * F.E[j]
        G.E[j]  = L[3, 1] * F.ρ[j] + L[3, 2] * F.ρu[j] + L[3, 3] * F.E[j]
    end
end

function project_to_realspace!(i, flux, flxrec)
    F_hat = flux.F_hat; G_hat = flux.G_hat; R = flxrec.R
    F_hat.ρ[i]  = R[1, 1] * G_hat.ρ[i] + R[1, 2] * G_hat.ρu[i] + R[1, 3] * G_hat.E[i]
    F_hat.ρu[i] = R[2, 1] * G_hat.ρ[i] + R[2, 2] * G_hat.ρu[i] + R[2, 3] * G_hat.E[i]
    F_hat.E[i]  = R[3, 1] * G_hat.ρ[i] + R[3, 2] * G_hat.ρu[i] + R[3, 3] * G_hat.E[i]
end

function plot_system(Q, grpar, filename)
    x = grpar.x; cr = grpar.cr_mesh
    plt = Plots.plot(x, Q.ρ, title="1D Euler equations", label="rho")
    Plots.plot!(x, Q.u, label="u")
    Plots.plot!(x, Q.P, label="P")
    display(plt)
    # Plots.png(plt, filename)
end

function euler(; γ=7/5, cfl=0.3, t_max=1.0)
    grpar = Weno.grid(size=256, min=-0.5, max=0.5)
    rkpar = Weno.preallocate_rungekutta_parameters(grpar)
    wepar = Weno.preallocate_weno_parameters(grpar)
    state = preallocate_statevectors(grpar)
    flux = preallocate_fluxes(grpar)
    flxrec = preallocate_fluxreconstruction(grpar)

    sod!(state.Q, grpar)
    # lax!(state.Q, grpar)
    # shu_osher!(state.Q, grpar)

    primitive_to_conserved!(state.Q, γ)
    t = 0.0; counter = 0

    while t < t_max
        update_physical_fluxes!(flux.F, state.Q)
        wepar.ev = max_eigval(state.Q, γ)
        dt = CFL_condition(wepar.ev, cfl, grpar)
        t += dt 
        
        # Component-wise reconstruction
        # for i in grpar.cr_cell
        #     update_local!(i, U, F, Q_local, F_local)
        #     F_hat.ρ[i]  = Weno.update_numerical_flux(Q_local.ρ,  F_local.ρ,  wepar)
        #     F_hat.ρu[i] = Weno.update_numerical_flux(Q_local.ρu, F_local.ρu, wepar)
        #     F_hat.E[i]  = Weno.update_numerical_flux(Q_local.E,  F_local.E,  wepar)
        # end

        # Characteristic-wise reconstruction
        for i in grpar.cr_cell
            update_jacobian!(i, state.Q, flxrec, γ, grpar)
            Weno.diagonalize_jacobian!(flxrec)
            project_to_localspace!(i, state, flux, flxrec)
            update_local!(i, state.Q_proj, flux.G, state.Q_local, flux.F_local)
            flux.G_hat.ρ[i]  = Weno.update_numerical_flux(state.Q_local.ρ,  flux.F_local.ρ,  wepar)
            flux.G_hat.ρu[i] = Weno.update_numerical_flux(state.Q_local.ρu, flux.F_local.ρu, wepar)
            flux.G_hat.E[i]  = Weno.update_numerical_flux(state.Q_local.E,  flux.F_local.E,  wepar)
            project_to_realspace!(i, flux, flxrec)
        end

        Weno.time_evolution!(state.Q.ρ,  flux.F_hat.ρ,  dt, grpar, rkpar) 
        Weno.time_evolution!(state.Q.ρu, flux.F_hat.ρu, dt, grpar, rkpar)
        Weno.time_evolution!(state.Q.E,  flux.F_hat.E,  dt, grpar, rkpar)

        conserved_to_primitive!(state.Q, γ)

        counter += 1
        if counter % 100 == 0
            @printf("Iteration %d: t = %2.3f, dt = %2.3e\n", counter, t, dt)
        end
    end

    @printf("%d iterations. t_max = %2.3f.\n", counter, t)
    plot_system(state.Q, grpar, "euler1d_shu_512")
end

# BenchmarkTools.@btime euler(t_max=0.14);
@time euler(t_max=0.14)

# Solves the 1D Euler equations
#   ∂Q/∂t + ∂F/∂x = 0
# where Q = [ρ, ρu, E],
#       F = [ρu, ρu² + P, u(E + P)]
# and   E = P/(γ-1) + 1/2 * ρu².
# For a gas with 5 dofs, γ = 7/5.

include("./Weno.jl")
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

struct SmoothnessFunctions{T}
    G₊::Vector{T}
    G₋::Vector{T}
end


function preallocate_statevectors(gridx)
    for x in [:ρ, :u, :P, :ρu, :E, :ρ2, :ρu2, :E2]
        @eval $x = zeros($gridx.nx)
    end
    for x in [:ρ3, :ρu3, :E3]
        @eval $x = zeros(6)
    end
    Q = Variables(ρ, u, P, ρu, E)
    Q_proj = ConservedVariables(ρ2, ρu2, E2)
    Q_local = ConservedVariables(ρ3, ρu3, E3)
    return StateVectors(Q, Q_proj, Q_local)
end

function preallocate_fluxes(gridx)
    for x in [:f1, :f2, :f3, :g1, :g2, :g3]
        @eval $x = zeros($gridx.nx)
    end
    for x in [:f1_hat, :f2_hat, :f3_hat, :g1_hat, :g2_hat, :g3_hat]
        @eval $x = zeros($gridx.nx+1)
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

function preallocate_fluxreconstruction(gridx)
    for x in [:ρ, :u, :P, :ρu, :E]
        @eval $x = zeros($gridx.nx+1)
    end
    Q_avg = Variables(ρ, u, P, ρu, E)
    J = zeros(3, 3)
    eval = zeros(3)
    evecL = zeros(3, 3)
    evecR = zeros(3, 3)
    return FluxReconstruction(Q_avg, J, eval, evecL, evecR)
end

function preallocate_smoothnessfunctions(gridx)
    G₊ = zeros(gridx.nx)
    G₋ = zeros(gridx.nx)
    return SmoothnessFunctions(G₊, G₋)
end

"""
Sod shock tube problem
x = [-0.5, 0.5], t_max = 0.14
"""
function sod!(Q, gridx)
    half = gridx.nx ÷ 2
    Q.ρ[1:half] .= 1; Q.ρ[half+1:end] .= 0.125
    Q.P[1:half] .= 1; Q.P[half+1:end] .= 0.1
end

"""
Lax problem
x = [-0.5, 0.5], t_max = 0.13
"""
function lax!(Q, gridx)
    half = gridx.nx ÷ 2
    Q.ρ[1:half] .= 0.445; Q.ρ[half+1:end] .= 0.5
    Q.u[1:half] .= 0.698
    Q.P[1:half] .= 3.528; Q.P[half+1:end] .= 0.571
end

"""
Shu-Osher problem
x = [-5, 5], t_max = 1.8
"""
function shu_osher!(Q, gridx)
    tenth = gridx.nx ÷ 10
    @. Q.ρ = 1.0 + 1/5 * sin(5 * gridx.x)
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

function arithmetic_average!(i, Q, Q_avg)
    Q_avg.ρ[i] = 1/2 * (Q.ρ[i] + Q.ρ[i+1])
    Q_avg.u[i] = 1/2 * (Q.u[i] + Q.u[i+1])
    Q_avg.P[i] = 1/2 * (Q.P[i] + Q.P[i+1])
end

sound_speed(P, ρ, γ) = sqrt(γ*P/ρ)

max_eigval(Q, γ) = @. $maximum(abs(Q.u) + sound_speed(Q.P, Q.ρ, γ))

CFL_condition(eigval, cfl, gridx) = 0.1 * cfl * gridx.dx / eigval

function update_physical_fluxes!(flux, Q)
    F = flux.F
    @. F.ρ  = Q.ρu
    @. F.ρu = Q.ρ * Q.u^2 + Q.P
    @. F.E  = Q.u * (Q.E + Q.P)
end

function update_smoothnessfunctions!(smooth, Q, α)
    @. smooth.G₊ = Q.ρ + Q.ρ * Q.u^2 + Q.P + α * Q.ρ * Q.u
    @. smooth.G₋ = Q.ρ + Q.ρ * Q.u^2 + Q.P - α * Q.ρ * Q.u
end

# J is defined starting from the leftmost j-1/2.
function update_jacobian!(i, Q, flxrec, γ)
    Q_avg = flxrec.Q_avg; J = flxrec.J
    arithmetic_average!(i, Q, Q_avg)

    ρ = Q_avg.ρ[i]; u = Q_avg.u[i]; P = Q_avg.P[i]

    J[2, 1] = -(3-γ)/2 * u^2
    J[3, 1] = (γ-2)/2 * u^3 - (γ*P/ρ) / (γ-1) * u
    J[1, 2] = 1
    J[2, 2] = (3-γ) * u
    J[3, 2] = (γ*P/ρ) / (γ-1) + (3-2γ)/2 * u^2
    J[2, 3] = γ-1
    J[3, 3] = γ * u
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

function update_local_smoothnessfunctions!(i, smooth, w)
    for j in 1:6
        w.fp[j] = smooth.G₊[i-3+j]
        w.fm[j] = smooth.G₋[i-3+j]
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

function time_evolution!(F̂, Q, gridx, dt, rkpar)
    Weno.weno_scheme!(F̂.ρ, gridx, rkpar)
    Weno.runge_kutta!(Q.ρ, dt, rkpar)

    Weno.weno_scheme!(F̂.ρu, gridx, rkpar)
    Weno.runge_kutta!(Q.ρu, dt, rkpar)

    Weno.weno_scheme!(F̂.E, gridx, rkpar)
    Weno.runge_kutta!(Q.E, dt, rkpar)
end

function plot_system(Q, gridx, filename)
    x = gridx.x; cr = gridx.cr_mesh
    plt = Plots.plot(x[cr], Q.ρ[cr], title="1D Euler equations", label="rho")
    Plots.plot!(x[cr], Q.u[cr], label="u")
    Plots.plot!(x[cr], Q.P[cr], label="P")
    display(plt)
    # Plots.png(plt, filename)
end

function euler(; γ=7/5, cfl=0.3, t_max=1.0)
    gridx = Weno.grid(size=1024, min=-0.5, max=0.5)
    rkpar = Weno.preallocate_rungekutta_parameters(gridx)
    wepar = Weno.preallocate_weno_parameters(gridx)
    state = preallocate_statevectors(gridx)
    flux = preallocate_fluxes(gridx)
    flxrec = preallocate_fluxreconstruction(gridx)
    smooth = preallocate_smoothnessfunctions(gridx)

    # sod!(state.Q, gridx)
    lax!(state.Q, gridx)
    # shu_osher!(state.Q, gridx)

    primitive_to_conserved!(state.Q, γ)
    t = 0.0; counter = 0; t0 = time()

    q = state.Q_local; f = flux.F_local
    F̂ = flux.F_hat; Ĝ = flux.G_hat
    while t < t_max
        update_physical_fluxes!(flux, state.Q)
        wepar.ev = max_eigval(state.Q, γ)
        dt = CFL_condition(wepar.ev, cfl, gridx)
        t += dt 
        
        # Component-wise reconstruction
        # for i in gridx.cr_cell
        #     update_local!(i, state.Q, flux.F, q, f)
        #     F̂.ρ[i]  = Weno.update_numerical_flux(q.ρ,  f.ρ,  wepar, ada=false)
        #     F̂.ρu[i] = Weno.update_numerical_flux(q.ρu, f.ρu, wepar, ada=false)
        #     F̂.E[i]  = Weno.update_numerical_flux(q.E,  f.E,  wepar, ada=false)
        # end

        # Characteristic-wise reconstruction
        # for i in gridx.cr_cell
        #     update_jacobian!(i, state.Q, flxrec, γ)
        #     Weno.diagonalize_jacobian!(flxrec)
        #     project_to_localspace!(i, state, flux, flxrec)
        #     update_local!(i, state.Q_proj, flux.G, q, f)
        #     Ĝ.ρ[i]  = Weno.update_numerical_flux(q.ρ,  f.ρ,  wepar, ada=false)
        #     Ĝ.ρu[i] = Weno.update_numerical_flux(q.ρu, f.ρu, wepar, ada=false)
        #     Ĝ.E[i]  = Weno.update_numerical_flux(q.E,  f.E,  wepar, ada=false)
        #     project_to_realspace!(i, flux, flxrec)
        # end

        # AdaWENO scheme
        update_smoothnessfunctions!(smooth, state.Q, wepar.ev)
        for i in gridx.cr_cell
            update_local_smoothnessfunctions!(i, smooth, wepar)
            Weno.nonlinear_weights_plus!(wepar)
            Weno.nonlinear_weights_minus!(wepar)
            Weno.update_switches!(wepar)
            if wepar.θp > 0.5 && wepar.θm > 0.5
                update_local!(i, state.Q, flux.F, q, f)
                F̂.ρ[i]  = Weno.update_numerical_flux(q.ρ,  f.ρ,  wepar, ada=true)
                F̂.ρu[i] = Weno.update_numerical_flux(q.ρu, f.ρu, wepar, ada=true)
                F̂.E[i]  = Weno.update_numerical_flux(q.E,  f.E,  wepar, ada=true)
            else
                update_jacobian!(i, state.Q, flxrec, γ)
                Weno.diagonalize_jacobian!(flxrec)
                project_to_localspace!(i, state, flux, flxrec)
                update_local!(i, state.Q_proj, flux.G, q, f)
                Ĝ.ρ[i]  = Weno.update_numerical_flux(q.ρ,  f.ρ,  wepar, ada=true)
                Ĝ.ρu[i] = Weno.update_numerical_flux(q.ρu, f.ρu, wepar, ada=true)
                Ĝ.E[i]  = Weno.update_numerical_flux(q.E,  f.E,  wepar, ada=true)
                project_to_realspace!(i, flux, flxrec)
            end
        end

        time_evolution!(flux.F_hat, state.Q, gridx, dt, rkpar)
        conserved_to_primitive!(state.Q, γ)

        counter += 1
        if counter % 200 == 0
            @printf("Iteration %d: t = %2.3f, dt = %2.3e, Elapsed time = %3.3f\n", 
                counter, t, dt, time() - t0)
        end
    end

    @printf("%d iterations. t_max = %2.3f. Elapsed time = %3.3f\n", 
        counter, t, time() - t0)
    plot_system(state.Q, gridx, "euler1d_shu_512")
end

# BenchmarkTools.@btime euler(t_max=0.13);
@time euler(t_max=0.13)

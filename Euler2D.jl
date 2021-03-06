include("./System.jl")
include("./Weno.jl")
using Printf, LinearAlgebra
import Plots, BenchmarkTools
Plots.pyplot()

abstract type BoundaryCondition end
struct DoubleMachReflection <: BoundaryCondition end
struct RayleighTaylor       <: BoundaryCondition end
struct RiemannNatural       <: BoundaryCondition end

struct Variables{T}
    ρ::T    # density
    u::T    # velocity
    v::T
    P::T    # pressure
    ρu::T   # momentum
    ρv::T
    E::T    # energy
end

struct ConservedVariables{T}
    ρ::T
    ρu::T
    ρv::T
    E::T
end

struct StateVectors{T}
    Q::Variables{Matrix{T}}                  # real space
    Q_proj::ConservedVariables{Matrix{T}}    # characteristic space
    Q_local::ConservedVariables{Vector{T}}   # local real space
end

struct Fluxes{T}
    Fx::ConservedVariables{Matrix{T}}        # physical flux, real space
    Fy::ConservedVariables{Matrix{T}}
    Gx::ConservedVariables{Matrix{T}}        # physical flux, characteristic space
    Gy::ConservedVariables{Matrix{T}}
    Fx_hat::ConservedVariables{Matrix{T}}    # numerical flux, real space
    Fy_hat::ConservedVariables{Matrix{T}}
    Gx_hat::ConservedVariables{Matrix{T}}    # numerical flux, characteristic space
    Gy_hat::ConservedVariables{Matrix{T}}
    F_local::ConservedVariables{Vector{T}}   # local physical flux, real space
end

mutable struct FluxReconstruction{T}
    Q_avg::Variables{Matrix{T}}   # averaged quantities
    Lx::Matrix{T}         # left eigenvectors
    Ly::Matrix{T}
    Rx::Matrix{T}         # right eigenvectors
    Ry::Matrix{T}
end

struct SmoothnessFunctions{T}
    G₊::Matrix{T}
    G₋::Matrix{T}
end


function preallocate_statevectors(sys)
    nx = sys.gridx.nx; ny = sys.gridy.nx
    for x in [:ρ, :u, :v, :P, :ρu, :ρv, :E, :ρ2, :ρu2, :ρv2, :E2]
        @eval $x = zeros($nx, $ny)
    end
    for x in [:ρ3, :ρu3, :ρv3, :E3]
        @eval $x = zeros(6)
    end
    Q = Variables(ρ, u, v, P, ρu, ρv, E)
    Q_proj = ConservedVariables(ρ2, ρu2, ρv2, E2)
    Q_local = ConservedVariables(ρ3, ρu3, ρv3, E3)
    return StateVectors(Q, Q_proj, Q_local)
end

function preallocate_fluxes(sys)
    nx = sys.gridx.nx; ny = sys.gridy.nx
    for x in [:fx1, :fx2, :fx3, :fx4, :fy1, :fy2, :fy3, :fy4,
              :gx1, :gx2, :gx3, :gx4, :gy1, :gy2, :gy3, :gy4]
        @eval $x = zeros($nx, $ny)
    end
    for x in [:fx1_hat, :fx2_hat, :fx3_hat, :fx4_hat, :fy1_hat, :fy2_hat, :fy3_hat, :fy4_hat,
              :gx1_hat, :gx2_hat, :gx3_hat, :gx4_hat, :gy1_hat, :gy2_hat, :gy3_hat, :gy4_hat]
        @eval $x = zeros($nx+1, $ny+1)
    end
    for x in [:f1_local, :f2_local, :f3_local, :f4_local]
        @eval $x = zeros(6)
    end
    Fx = ConservedVariables(fx1, fx2, fx3, fx4)
    Fy = ConservedVariables(fy1, fy2, fy3, fy4)
    Gx = ConservedVariables(gx1, gx2, gx3, gx4)
    Gy = ConservedVariables(gy1, gy2, gy3, gy4)
    Fx_hat = ConservedVariables(fx1_hat, fx2_hat, fx3_hat, fx4_hat)
    Fy_hat = ConservedVariables(fy1_hat, fy2_hat, fy3_hat, fy4_hat)
    Gx_hat = ConservedVariables(gx1_hat, gx2_hat, gx3_hat, gx4_hat)
    Gy_hat = ConservedVariables(gy1_hat, gy2_hat, gy3_hat, gy4_hat)
    F_local = ConservedVariables(f1_local, f2_local, f3_local, f4_local)
    return Fluxes(Fx, Fy, Gx, Gy, Fx_hat, Fy_hat, Gx_hat, Gy_hat, F_local)
end

function preallocate_fluxreconstruction(sys)
    nx = sys.gridx.nx; ny = sys.gridy.nx
    for x in [:ρ, :u, :v, :P, :ρu, :ρv, :E]
        @eval $x = zeros($nx+1, $ny+1)
    end
    Q_avg = Variables(ρ, u, v, P, ρu, ρv, E)
    for x in [:evecLx, :evecLy, :evecRx, :evecRy]
        @eval $x = zeros(4, 4)
    end
    return FluxReconstruction(Q_avg, evecLx, evecLy, evecRx, evecRy)
end

function preallocate_smoothnessfunctions(sys)
    nx = sys.gridx.nx; ny = sys.gridy.nx
    G₊ = zeros(nx, ny)
    G₋ = zeros(nx, ny)
    return SmoothnessFunctions(G₊, G₋)
end

"""
Case 6 Riemann problem
(x, y) = [0, 1] × [0, 1], t_max = 0.3
"""
function case6!(Q, sys)
    halfx = sys.gridx.nx ÷ 2; halfy = sys.gridy.nx ÷ 2

    Q.ρ[1:halfx, 1:halfy] .= 1.0
    Q.u[1:halfx, 1:halfy] .= -0.75
    Q.v[1:halfx, 1:halfy] .= 0.5
    Q.P[1:halfx, 1:halfy] .= 1.0
    Q.ρ[halfx+1:end, 1:halfy] .= 3.0
    Q.u[halfx+1:end, 1:halfy] .= -0.75
    Q.v[halfx+1:end, 1:halfy] .= -0.5
    Q.P[halfx+1:end, 1:halfy] .= 1.0
    Q.ρ[1:halfx, halfy+1:end] .= 2.0
    Q.u[1:halfx, halfy+1:end] .= 0.75
    Q.v[1:halfx, halfy+1:end] .= 0.5
    Q.P[1:halfx, halfy+1:end] .= 1.0
    Q.ρ[halfx+1:end, halfy+1:end] .= 1.0
    Q.u[halfx+1:end, halfy+1:end] .= 0.75
    Q.v[halfx+1:end, halfy+1:end] .= -0.5
    Q.P[halfx+1:end, halfy+1:end] .= 1.0
end

"""
Case 12 Riemann problem
(x, y) = [0, 1] × [0, 1], t_max = 0.25
"""
function case12!(Q, sys)
    halfx = sys.gridx.nx ÷ 2; halfy = sys.gridy.nx ÷ 2

    Q.ρ[1:halfx, 1:halfy] .= 0.8
    Q.P[1:halfx, 1:halfy] .= 1.0
    Q.ρ[halfx+1:end, 1:halfy] .= 1.0
    Q.v[halfx+1:end, 1:halfy] .= 0.7276
    Q.P[halfx+1:end, 1:halfy] .= 1.0
    Q.ρ[1:halfx, halfy+1:end] .= 1.0
    Q.u[1:halfx, halfy+1:end] .= 0.7276
    Q.P[1:halfx, halfy+1:end] .= 1.0
    Q.ρ[halfx+1:end, halfy+1:end] .= 0.5313
    Q.P[halfx+1:end, halfy+1:end] .= 0.4
end

"""
Double Mach reflection problem
(x, y) = [0, 3.25] × [0, 1], t_max = 0.2
"""
function doublemach!(Q, sys)
    nx = sys.gridx.nx; ny = sys.gridy.nx; γ = sys.γ
    for j in 1:ny, i in 1:nx
        x = sys.gridx.x[i]; y = sys.gridy.x[j]
        if y < sqrt(3) * (x - 1/6)
            Q.ρ[i, j] = 1.4
            Q.P[i, j] = 1.0
        else
            Q.ρ[i, j] = 8.0
            Q.u[i, j] = 4.125 * sqrt(3)
            Q.v[i, j] = -4.125
            Q.P[i, j] = 116.5
        end
    end
end

"""
Rayleigh-Taylor instability bubble
(x, y) = [0, 1/6] × [0, 1], t_max = 8.5
Interface at y = 1/2 + 0.01*cos(6πx)
"""
function rayleightaylor!(Q, sys)

end

function primitive_to_conserved!(Q, γ)
    @. Q.ρu = Q.ρ * Q.u
    @. Q.ρv = Q.ρ * Q.v
    @. Q.E  = Q.P / (γ-1) + 1/2 * Q.ρ * (Q.u^2 + Q.v^2)
end

function conserved_to_primitive!(Q, γ)
    @. Q.u = Q.ρu / Q.ρ
    @. Q.v = Q.ρv / Q.ρ
    @. Q.P = (γ-1) * (Q.E - (Q.ρu^2 + Q.ρv^2) / 2Q.ρ)
end

function arithmetic_average!(i, j, Q, Q_avg, dim)
    if dim == :X
        Q_avg.ρ[i, j] = 1/2 * (Q.ρ[i, j] + Q.ρ[i+1, j])
        Q_avg.u[i, j] = 1/2 * (Q.u[i, j] + Q.u[i+1, j])
        Q_avg.v[i, j] = 1/2 * (Q.v[i, j] + Q.v[i+1, j])
        Q_avg.P[i, j] = 1/2 * (Q.P[i, j] + Q.P[i+1, j])
        Q_avg.E[i, j] = 1/2 * (Q.E[i, j] + Q.E[i+1, j])
    elseif dim == :Y
        Q_avg.ρ[i, j] = 1/2 * (Q.ρ[i, j] + Q.ρ[i, j+1])
        Q_avg.u[i, j] = 1/2 * (Q.u[i, j] + Q.u[i, j+1])
        Q_avg.v[i, j] = 1/2 * (Q.v[i, j] + Q.v[i, j+1])
        Q_avg.P[i, j] = 1/2 * (Q.P[i, j] + Q.P[i, j+1])
        Q_avg.E[i, j] = 1/2 * (Q.E[i, j] + Q.E[i, j+1])
    end
end

sound_speed(P, ρ, γ) = sqrt(γ*P/ρ)

max_speed(Q, γ) = @. $maximum(sqrt(Q.u^2 + Q.v^2) + sound_speed(Q.P, Q.ρ, γ))

function CFL_condition(v, cfl, sys)
    dx = sys.gridx.dx; dy = sys.gridy.dx
    return 0.1 * cfl * dx * dy / (v * (dx + dy))
end

function update_physical_fluxes!(flux, Q)
    Fx = flux.Fx; Fy = flux.Fy 
    @. Fx.ρ  = Q.ρu
    @. Fx.ρu = Q.ρ * Q.u^2 + Q.P
    @. Fx.ρv = Q.ρ * Q.u * Q.v
    @. Fx.E  = Q.u * (Q.E + Q.P)
    @. Fy.ρ  = Q.ρv
    @. Fy.ρu = Q.ρ * Q.u * Q.v
    @. Fy.ρv = Q.ρ * Q.v^2 + Q.P
    @. Fy.E  = Q.v * (Q.E + Q.P)
end

function update_smoothnessfunctions!(smooth, Q, α)
    @. smooth.G₊ = Q.ρ + Q.E + α * Q.ρ * (Q.u + Q.v)
    @. smooth.G₋ = Q.ρ + Q.E - α * Q.ρ * (Q.u + Q.v)
end

function update_xeigenvectors!(i, j, Q, flxrec, γ)
    R = flxrec.Rx; Q_avg = flxrec.Q_avg
    arithmetic_average!(i, j, Q, Q_avg, :X)

    ρ = Q_avg.ρ[i, j]; u = Q_avg.u[i, j]
    v = Q_avg.v[i, j]; P = Q_avg.P[i, j]
    E = Q_avg.E[i, j]
    c = sound_speed(P, ρ, γ)

    R[1, 1] = 1
    R[1, 2] = 1
    R[1, 4] = 1

    R[2, 1] = u - c
    R[2, 2] = u
    R[2, 4] = u + c

    R[3, 1] = v
    R[3, 2] = v
    R[3, 3] = 1
    R[3, 4] = v

    R[4, 1] = (E + P) / ρ - c * u
    R[4, 2] = 1/2 * (u^2 + v^2)
    R[4, 3] = v
    R[4, 4] = (E + P) / ρ + c * u

    flxrec.Lx = inv(R)
end

function update_yeigenvectors!(i, j, Q, flxrec, γ)
    R = flxrec.Ry; Q_avg = flxrec.Q_avg
    arithmetic_average!(i, j, Q, Q_avg, :Y)

    ρ = Q_avg.ρ[i, j]; u = Q_avg.u[i, j]
    v = Q_avg.v[i, j]; P = Q_avg.P[i, j]
    E = Q_avg.E[i, j]
    c = sound_speed(P, ρ, γ)

    R[1, 1] = 1
    R[1, 2] = 1
    R[1, 4] = 1

    R[2, 1] = u
    R[2, 2] = u
    R[2, 3] = 1
    R[2, 4] = u

    R[3, 1] = v - c
    R[3, 2] = v
    R[3, 4] = v + c

    R[4, 1] = (E + P) / ρ - c * v
    R[4, 2] = 1/2 * (u^2 + v^2)
    R[4, 3] = u
    R[4, 4] = (E + P) / ρ + c * v

    flxrec.Ly = inv(R)
end

function update_local!(i, j, Q, F, Q_local, F_local, dim)
    if dim == :X
        for k in 1:6
            Q_local.ρ[k]  = Q.ρ[i-3+k, j]
            Q_local.ρu[k] = Q.ρu[i-3+k, j]
            Q_local.ρv[k] = Q.ρv[i-3+k, j]
            Q_local.E[k]  = Q.E[i-3+k, j]
            F_local.ρ[k]  = F.ρ[i-3+k, j]
            F_local.ρu[k] = F.ρu[i-3+k, j]
            F_local.ρv[k] = F.ρv[i-3+k, j]
            F_local.E[k]  = F.E[i-3+k, j]
        end
    elseif dim == :Y
        for k in 1:6
            Q_local.ρ[k]  = Q.ρ[i, j-3+k]
            Q_local.ρu[k] = Q.ρu[i, j-3+k]
            Q_local.ρv[k] = Q.ρv[i, j-3+k]
            Q_local.E[k]  = Q.E[i, j-3+k]
            F_local.ρ[k]  = F.ρ[i, j-3+k]
            F_local.ρu[k] = F.ρu[i, j-3+k]
            F_local.ρv[k] = F.ρv[i, j-3+k]
            F_local.E[k]  = F.E[i, j-3+k]
        end
    end
end

function update_local_smoothnessfunctions!(i, j, smooth, w, dim)
    if dim == :X
        for k in 1:6
            w.fp[k] = smooth.G₊[i-3+k, j]
            w.fm[k] = smooth.G₋[i-3+k, j]
        end
    elseif dim == :Y
        for k in 1:6
            w.fp[k] = smooth.G₊[i, j-3+k]
            w.fm[k] = smooth.G₋[i, j-3+k]
        end
    end
end

function project_to_localspace!(i, j, state, flux, flxrec, dim)
    Q = state.Q; Q_proj = state.Q_proj

    if dim == :X
        F = flux.Fx; G = flux.Gx; L = flxrec.Lx
        for k in i-2:i+3
            Qρ = Q.ρ[k, j]; Qρu = Q.ρu[k, j]; Qρv = Q.ρv[k, j]; QE = Q.E[k, j]
            Fρ = F.ρ[k, j]; Fρu = F.ρu[k, j]; Fρv = F.ρv[k, j]; FE = F.E[k, j]

            Q_proj.ρ[k, j]  = L[1, 1] * Qρ + L[1, 2] * Qρu + L[1, 3] * Qρv + L[1, 4] * QE
            Q_proj.ρu[k, j] = L[2, 1] * Qρ + L[2, 2] * Qρu + L[2, 3] * Qρv + L[2, 4] * QE
            Q_proj.ρv[k, j] = L[3, 1] * Qρ + L[3, 2] * Qρu + L[3, 3] * Qρv + L[3, 4] * QE
            Q_proj.E[k, j]  = L[4, 1] * Qρ + L[4, 2] * Qρu + L[3, 4] * Qρv + L[4, 4] * QE
            G.ρ[k, j]  = L[1, 1] * Fρ + L[1, 2] * Fρu + L[1, 3] * Fρv + L[1, 4] * FE
            G.ρu[k, j] = L[2, 1] * Fρ + L[2, 2] * Fρu + L[2, 3] * Fρv + L[2, 4] * FE
            G.ρv[k, j] = L[3, 1] * Fρ + L[3, 2] * Fρu + L[3, 3] * Fρv + L[3, 4] * FE
            G.E[k, j]  = L[4, 1] * Fρ + L[4, 2] * Fρu + L[4, 3] * Fρv + L[4, 4] * FE
        end
    elseif dim == :Y
        F = flux.Fy; G = flux.Gy; L = flxrec.Ly
        for k in j-2:j+3
            Qρ = Q.ρ[i, k]; Qρu = Q.ρu[i, k]; Qρv = Q.ρv[i, k]; QE = Q.E[i, k]
            Fρ = F.ρ[i, k]; Fρu = F.ρu[i, k]; Fρv = F.ρv[i, k]; FE = F.E[i, k]

            Q_proj.ρ[i, k]  = L[1, 1] * Qρ + L[1, 2] * Qρu + L[1, 3] * Qρv + L[1, 4] * QE
            Q_proj.ρu[i, k] = L[2, 1] * Qρ + L[2, 2] * Qρu + L[2, 3] * Qρv + L[2, 4] * QE
            Q_proj.ρv[i, k] = L[3, 1] * Qρ + L[3, 2] * Qρu + L[3, 3] * Qρv + L[3, 4] * QE
            Q_proj.E[i, k]  = L[4, 1] * Qρ + L[4, 2] * Qρu + L[3, 4] * Qρv + L[4, 4] * QE
            G.ρ[i, k]  = L[1, 1] * Fρ + L[1, 2] * Fρu + L[1, 3] * Fρv + L[1, 4] * FE
            G.ρu[i, k] = L[2, 1] * Fρ + L[2, 2] * Fρu + L[2, 3] * Fρv + L[2, 4] * FE
            G.ρv[i, k] = L[3, 1] * Fρ + L[3, 2] * Fρu + L[3, 3] * Fρv + L[3, 4] * FE
            G.E[i, k]  = L[4, 1] * Fρ + L[4, 2] * Fρu + L[4, 3] * Fρv + L[4, 4] * FE
        end
    end
end

function project_to_realspace!(i, j, flux, flxrec, dim)
    if dim == :X
        F̂ = flux.Fx_hat; Ĝ = flux.Gx_hat; R = flxrec.Rx
    elseif dim == :Y
        F̂ = flux.Fy_hat; Ĝ = flux.Gy_hat; R = flxrec.Ry
    end
    Ĝρ = Ĝ.ρ[i, j]; Ĝρu = Ĝ.ρu[i, j]; Ĝρv = Ĝ.ρv[i, j]; ĜE = Ĝ.E[i, j]

    F̂.ρ[i, j]  = R[1, 1] * Ĝρ + R[1, 2] * Ĝρu + R[1, 3] * Ĝρv + R[1, 4] * ĜE
    F̂.ρu[i, j] = R[2, 1] * Ĝρ + R[2, 2] * Ĝρu + R[2, 3] * Ĝρv + R[2, 4] * ĜE
    F̂.ρv[i, j] = R[3, 1] * Ĝρ + R[3, 2] * Ĝρu + R[3, 3] * Ĝρv + R[3, 4] * ĜE
    F̂.E[i, j]  = R[4, 1] * Ĝρ + R[4, 2] * Ĝρu + R[4, 3] * Ĝρv + R[4, 4] * ĜE
end

function update_numerical_fluxes!(i, j, F̂, q, f, wepar, ada)
    F̂.ρ[i, j]  = Weno.update_numerical_flux(q.ρ,  f.ρ,  wepar, ada)
    F̂.ρu[i, j] = Weno.update_numerical_flux(q.ρu, f.ρu, wepar, ada)
    F̂.ρv[i, j] = Weno.update_numerical_flux(q.ρv, f.ρv, wepar, ada)
    F̂.E[i, j]  = Weno.update_numerical_flux(q.E,  f.E,  wepar, ada)
end

function time_evolution!(F̂x, F̂y, Q, sys, dt, rkpar)
    gridx = sys.gridx; gridy = sys.gridy

    Weno.weno_scheme!(F̂x.ρ, F̂y.ρ, gridx, gridy, rkpar)
    Weno.runge_kutta!(Q.ρ, dt, rkpar)

    Weno.weno_scheme!(F̂x.ρu, F̂y.ρu, gridx, gridy, rkpar)
    Weno.runge_kutta!(Q.ρu, dt, rkpar)

    Weno.weno_scheme!(F̂x.ρv, F̂y.ρv, gridx, gridy, rkpar)
    Weno.runge_kutta!(Q.ρv, dt, rkpar)

    Weno.weno_scheme!(F̂x.E, F̂y.E, gridx, gridy, rkpar)
    Weno.runge_kutta!(Q.E, dt, rkpar)
end

"""
Natural boundary conditions for the Riemann problems assigns the ghost points values
that lie directly outside the boundaries (zero gradient).
"""
function boundary_conditions!(Q, sys, bctype::RiemannNatural)
    for n in 1:3, i in sys.gridx.cr_mesh
        Q.ρ[i, n]  = Q.ρ[i, 4];  Q.ρ[i, end-n+1]  = Q.ρ[i, end-3]
        Q.ρu[i, n] = Q.ρu[i, 4]; Q.ρu[i, end-n+1] = Q.ρu[i, end-3]
        Q.ρv[i, n] = Q.ρv[i, 4]; Q.ρv[i, end-n+1] = Q.ρv[i, end-3]
        Q.E[i, n]  = Q.E[i, 4];  Q.E[i, end-n+1]  = Q.E[i, end-3]
    end
    for i in sys.gridy.cr_mesh, n in 1:3
        Q.ρ[n, i]  = Q.ρ[4, i];  Q.ρ[end-n+1, i]  = Q.ρ[end-3, i]
        Q.ρu[n, i] = Q.ρu[4, i]; Q.ρu[end-n+1, i] = Q.ρu[end-3, i]
        Q.ρv[n, i] = Q.ρv[4, i]; Q.ρv[end-n+1, i] = Q.ρv[end-3, i]
        Q.E[n, i]  = Q.E[4, i];  Q.E[end-n+1, i]  = Q.E[end-3, i]
    end
end

"""
Reflecting boundary condition on the bottom at x = [1/6, 1], natural on the sides.
On the top, the values are set to describe (preserve) the shock.
"""
function boundary_conditions!(Q, sys, bctype::DoubleMachReflection)
    onesixth = argmin(@. abs(sys.gridx.x - 1/6))
    shockfront = 0
    for i in sys.gridx.cr_mesh
        if abs(Q.ρ[i, end-3] - 8.0) > 0.1
            shockfront = i-1
            break
        end
    end

    # x = [0, 1/6], y = 0
    for n in 1:3, i in 1:onesixth
        Q.ρ[i, n]  = Q.ρ[i, 4]
        Q.ρu[i, n] = Q.ρu[i, 4]
        Q.ρv[i, n] = Q.ρv[i, 4]
        Q.E[i, n]  = Q.E[i, 4]
    end
    # x = [1/6, 1], y = 0
    for n in 1:3, i in onesixth+1:sys.gridx.nx
        Q.ρ[i, n]  = Q.ρ[i, 7-n]
        Q.ρu[i, n] = -Q.ρu[i, 7-n]
        Q.ρv[i, n] = -Q.ρv[i, 7-n]
        Q.E[i, n]  = Q.E[i, 7-n]
    end
    for n in 1:3, i in sys.gridx.cr_mesh
        Q.ρ[i, end-n+1]  = Q.ρ[i, end-3]
        Q.ρu[i, end-n+1] = Q.ρu[i, end-3]
        Q.ρv[i, end-n+1] = Q.ρv[i, end-3]
        Q.E[i, end-n+1]  = Q.E[i, end-3]
    end
    # Propagates shock into boundary at oblique angle as in the computational range
    for n in 1:3
        Q.ρ[shockfront+1, end-n+1]  = Q.ρ[shockfront, end-3]
        Q.ρu[shockfront+1, end-n+1] = Q.ρu[shockfront, end-3]
        Q.ρv[shockfront+1, end-n+1] = Q.ρv[shockfront, end-3]
        Q.E[shockfront+1, end-n+1]  = Q.E[shockfront, end-3]
    end
    Q.ρ[shockfront+2, end]  = Q.ρ[shockfront, end-3]
    Q.ρu[shockfront+2, end] = Q.ρu[shockfront, end-3]
    Q.ρv[shockfront+2, end] = Q.ρv[shockfront, end-3]
    Q.E[shockfront+2, end]  = Q.E[shockfront, end-3]
    
    for i in sys.gridy.cr_mesh, n in 1:3
        Q.ρ[n, i]  = Q.ρ[4, i];  Q.ρ[end-n+1, i]  = Q.ρ[end-3, i]
        Q.ρu[n, i] = Q.ρu[4, i]; Q.ρu[end-n+1, i] = Q.ρu[end-3, i]
        Q.ρv[n, i] = Q.ρv[4, i]; Q.ρv[end-n+1, i] = Q.ρv[end-3, i]
        Q.E[n, i]  = Q.E[4, i];  Q.E[end-n+1, i]  = Q.E[end-3, i]
    end
end

function boundary_conditions!(Q, sys, bctype::RayleighTaylor)

end

function plot_system(q, sys, titlename, filename)
    crx = sys.gridx.x[sys.gridx.cr_mesh]; cry = sys.gridy.x[sys.gridy.cr_mesh]
    q_transposed = q[sys.gridx.cr_mesh, sys.gridy.cr_mesh] |> transpose
    plt = Plots.contour(crx, cry, q_transposed, title=titlename, 
                        fill=false, linecolor=:plasma, levels=30, aspect_ratio=1.0)
    display(plt)
    # Plots.pdf(plt, filename)
end

function euler(; γ=7/5, cfl=0.6, t_max=1.0)
    gridx = grid(size=128, min=0.0, max=1.0)
    gridy = grid(size=128, min=0.0, max=1.0)
    sys = SystemParameters2D(gridx, gridy, 2, 4, γ)
    rkpar = Weno.preallocate_rungekutta_parameters(gridx, gridy)
    wepar = Weno.preallocate_weno_parameters()
    state = preallocate_statevectors(sys)
    flux = preallocate_fluxes(sys)
    flxrec = preallocate_fluxreconstruction(sys)
    smooth = preallocate_smoothnessfunctions(sys)

    # doublemach!(state.Q, sys)
    # rayleightaylor!(state.Q, sys)
    case6!(state.Q, sys)
    # case12!(state.Q, sys)
    
    primitive_to_conserved!(state.Q, γ)
    boundary_conditions!(state.Q, sys, RiemannNatural())
    t = 0.0; counter = 0; t0 = time()
    # plot_system(state.Q.ρ, sys, "Mass density", "case12_rho_512x512_ada")

    q = state.Q_local; f = flux.F_local
    while t < t_max
        update_physical_fluxes!(flux, state.Q)

        wepar.ev = max_speed(state.Q, γ)
        dt = CFL_condition(wepar.ev, cfl, sys)
        t += dt 
        
        # Component-wise reconstruction
        # for j in gridy.cr_cell, i in gridx.cr_cell
        #     update_local!(i, j, state.Q, flux.Fx, q, f, :X)
        #     update_numerical_fluxes!(i, j, flux.Fx_hat, q, f, wepar, false)
        # end
        # for j in gridy.cr_cell, i in gridx.cr_cell
        #     update_local!(i, j, state.Q, flux.Fy, q, f, :Y)
        #     update_numerical_fluxes!(i, j, flux.Fy_hat, q, f, wepar, false)
        # end

        # Characteristic-wise reconstruction
        # for j in gridy.cr_cell, i in gridx.cr_cell
        #     update_xeigenvectors!(i, j, state.Q, flxrec, γ)
        #     project_to_localspace!(i, j, state, flux, flxrec, :X)
        #     update_local!(i, j, state.Q_proj, flux.Gx, q, f, :X)
        #     update_numerical_fluxes!(i, j, flux.Gx_hat, q, f, wepar, false)
        #     project_to_realspace!(i, j, flux, flxrec, :X)
        # end
        # for j in gridy.cr_cell, i in gridx.cr_cell
        #     update_yeigenvectors!(i, j, state.Q, flxrec, γ)
        #     project_to_localspace!(i, j, state, flux, flxrec, :Y)
        #     update_local!(i, j, state.Q_proj, flux.Gy, q, f, :Y)
        #     update_numerical_fluxes!(i, j, flux.Gy_hat, q, f, wepar, false)
        #     project_to_realspace!(i, j, flux, flxrec, :Y)
        # end

        # AdaWENO scheme
        update_smoothnessfunctions!(smooth, state.Q, wepar.ev)
        for j in gridy.cr_cell, i in gridx.cr_cell
            update_local_smoothnessfunctions!(i, j, smooth, wepar, :X)
            Weno.nonlinear_weights_plus!(wepar)
            Weno.nonlinear_weights_minus!(wepar)
            Weno.update_switches!(wepar)
            if wepar.θp > 0.5 && wepar.θm > 0.5
                update_local!(i, j, state.Q, flux.Fx, q, f, :X)
                update_numerical_fluxes!(i, j, flux.Fx_hat, q, f, wepar, true)
            else
                update_xeigenvectors!(i, j, state.Q, flxrec, γ)
                project_to_localspace!(i, j, state, flux, flxrec, :X)
                update_local!(i, j, state.Q_proj, flux.Gx, q, f, :X)
                update_numerical_fluxes!(i, j, flux.Gx_hat, q, f, wepar, false)
                project_to_realspace!(i, j, flux, flxrec, :X)
            end
        end
        for j in gridy.cr_cell, i in gridx.cr_cell
            update_local_smoothnessfunctions!(i, j, smooth, wepar, :Y)
            Weno.nonlinear_weights_plus!(wepar)
            Weno.nonlinear_weights_minus!(wepar)
            Weno.update_switches!(wepar)
            if wepar.θp > 0.5 && wepar.θm > 0.5
                update_local!(i, j, state.Q, flux.Fy, q, f, :Y)
                update_numerical_fluxes!(i, j, flux.Fy_hat, q, f, wepar, true)
            else
                update_yeigenvectors!(i, j, state.Q, flxrec, γ)
                project_to_localspace!(i, j, state, flux, flxrec, :Y)
                update_local!(i, j, state.Q_proj, flux.Gy, q, f, :Y)
                update_numerical_fluxes!(i, j, flux.Gy_hat, q, f, wepar, false)
                project_to_realspace!(i, j, flux, flxrec, :Y)
            end
        end

        time_evolution!(flux.Fx_hat, flux.Fy_hat, state.Q, sys, dt, rkpar)
        boundary_conditions!(state.Q, sys, RiemannNatural())
        conserved_to_primitive!(state.Q, γ)

        counter += 1
        if counter % 100 == 0
            @printf("Iteration %d: t = %2.3f, dt = %2.3e, elapsed = %3.3f\n", 
                counter, t, dt, time() - t0)
            # plot_system(state.Q.ρ, sys, "Mass density", "case12_rho_512x512_ada")
        end
        # if counter == 20 return end
    end

    @printf("%d iterations. t_max = %2.3f. Elapsed time = %3.3f\n", 
        counter, t, time() - t0)
    plot_system(state.Q.ρ, sys, "Mass density", "mach_rho_832x512_ada")
    plot_system(state.Q.P, sys, "Pressure", "mach_P_832x512_ada")
end

# BenchmarkTools.@btime euler(t_max=0.01);
@time euler(t_max=0.3)

include("./System.jl")
include("./Weno.jl")
using Printf, LinearAlgebra
import Plots, BenchmarkTools
Plots.pyplot()

abstract type BoundaryCondition end
struct Periodic <: BoundaryCondition end

"""
Variables: (ρ, u, v, w, P, ρu, ρv, ρw, E, Bx, By, Bz, Az)
Conserved: (ρ, ρu, ρv, ρw, E, Bx, By, Bz)
Dynamics:  (ρ, ρu, ρv, ρw, E, Bx, By, Bz, Az)
"""
struct StateVectors{T}
    Q::Array{T, 3}        # real space
    Q_proj::Array{T, 3}   # characteristic space
    Q_local::Matrix{T}    # local real space
    A_proj::Matrix{T}     # characteristic space vector potential
    A_local::Vector{T}    # local real space vector potential
end

struct Fluxes{T}
    Fx::Array{T, 3}       # physical flux, real space
    Fy::Array{T, 3}
    Gx::Array{T, 3}       # physical flux, characteristic space
    Gy::Array{T, 3}
    Fx_hat::Array{T, 3}   # numerical flux, real space
    Fy_hat::Array{T, 3}
    Gx_hat::Array{T, 3}   # numerical flux, characteristic space
    Gy_hat::Array{T, 3}
    F_local::Matrix{T}    # local physical flux, real space
end

mutable struct FluxReconstruction{T}
    Q_avg::Array{T, 3}   # averaged quantities
    A_avg::Matrix{T}     # averaged vector potential
    Lx::Matrix{T}        # left eigenvectors
    Ly::Matrix{T}
    Rx::Matrix{T}        # right eigenvectors
    Ry::Matrix{T}
end

struct SmoothnessFunctions{T}
    G₊::Matrix{T}
    G₋::Matrix{T}
end


function preallocate_statevectors(sys)
    nx = sys.gridx.nx; ny = sys.gridy.nx
    Q = zeros(nx, ny, 13)
    Q_proj = zeros(nx, ny, 8)
    Q_local = zeros(6, 8)
    A_proj = zeros(nx, ny)
    A_local = zeros(6)
    return StateVectors(Q, Q_proj, Q_local, A_proj, A_local)
end

function preallocate_fluxes(sys)
    nx = sys.gridx.nx; ny = sys.gridy.nx
    for x in [:Fx, :Fy, :Gx, :Gy]
        @eval $x = zeros($nx, $ny, 8)
    end
    for x in [:Fx_hat, :Fy_hat, :Gx_hat, :Gy_hat]
        @eval $x = zeros($nx+1, $ny+1, 8)
    end
    F_local = zeros(6, 8)

    return Fluxes(Fx, Fy, Gx, Gy, Fx_hat, Fy_hat, Gx_hat, Gy_hat, F_local)
end

function preallocate_fluxreconstruction(sys)
    nx = sys.gridx.nx; ny = sys.gridy.nx
    Q_avg = zeros(nx, ny, 13)
    for x in [:evecLx, :evecLy, :evecRx, :evecRy]
        @eval $x = zeros(8, 8)
    end
    return FluxReconstruction(Q_avg, evecLx, evecLy, evecRx, evecRy)
end

function preallocate_smoothnessfunctions(gridx, gridy)
    G₊ = zeros(gridx.nx, gridy.nx)
    G₋ = zeros(gridx.nx, gridy.nx)
    return SmoothnessFunctions(G₊, G₋)
end

mag2(x, y, z) = x^2 + y^2 + z^2

"""
Orszag-Tang vortex
(x, y) = [0, 2π] × [0, 2π], t_max = 4.0
"""
function orszagtang!(Q, sys)
    nx = sys.gridx.nx; ny = sys.gridy.nx; γ = sys.γ
    for j in ny, i in nx
        x = sys.gridx.x[i]; y = sys.gridy.x[j]
        Q[i, j, 1] = γ^2
        Q[i, j, 2] = -sin(y)
        Q[i, j, 3] = sin(x)
        Q[i, j, 5] = γ
        Q[i, j, 10] = -sin(y)
        Q[i, j, 11] = sin(2x)
        Q[i, j, 13] = 1/2 * cos(2x) + cos(y)
    end
end

function primitive_to_conserved!(Q, sys)
    nx = sys.gridx.nx; ny = sys.gridy.nx; γ = sys.γ
    for j in ny, i in nx
        Q[i, j, 6] = Q[i, j, 1] * Q[i, j, 2]
        Q[i, j, 7] = Q[i, j, 1] * Q[i, j, 3]
        Q[i, j, 8] = Q[i, j, 1] * Q[i, j, 4]
        Q[i, j, 9] = Q[i, j, 5] / (γ-1) + 
                     1/2 * Q[i, j, 1] * (mag2(Q[i, j, 2], Q[i, j, 3], Q[i, j, 4])) +
                     1/8π * mag2(Q[i, j, 10], Q[i, j, 11], Q[i, j, 12])
    end
end

function conserved_to_primitive!(Q, sys)
    nx = sys.gridx.nx; ny = sys.gridy.nx; γ = sys.γ
    for j in ny, i in nx
        Q[i, j, 2] = Q[i, j, 6] / Q[i, j, 1]
        Q[i, j, 3] = Q[i, j, 7] / Q[i, j, 1]
        Q[i, j, 4] = Q[i, j, 8] / Q[i, j, 1]
        Q[i, j, 5] = (γ-1) * ( Q[i, j, 9] - 
                               mag2(Q[i, j, 6], Q[i, j, 7], Q[i, j, 8]) / 2Q[i, j, 1] -
                               1/8π * mag2(Q[i, j, 10], Q[i, j, 11], Q[i, j, 12]) )
    end
end

function arithmetic_average!(i, j, Q, Q_avg, dim)
    if dim == :X
        for k in 1:13
            Q_avg[i, j, k] = 1/2 * (Q[i, j, k] + Q[i+1, j, k])
        end
    elseif dim == :Y
        for k in 1:13
            Q_avg[i, j, k] = 1/2 * (Q[i, j, k] + Q[i, j+1, k])
        end
    end
end

"""
Magnetosonic wave
U_m = sqrt(U_A^2 + U_s^2), where
U_A is the Alfven velocity and U_s is the sound velocity
"""
magnetosonic(ρ, P, Bx, By, Bz, γ) = sqrt(mag2(Bx, By, Bz) / (4π*ρ) + γ*P/ρ)

"""
CFL condition
U_max = max(U + U_m), where
U = sqrt(u^2 + v^2 + w^2) is the fluid velocity
"""

function max_eigval(Q, γ) 
    return @views @. $maximum( sqrt(mag2(Q[:,:,2], Q[:,:,3], Q[:,:,4])) + 
                               magnetosonic(Q[:,:,1], Q[:,:,5], Q[:,:,10], Q[:,:,11], Q[:,:,12], γ) )
end 

function CFL_condition(v, cfl, sys)
    dx = sys.gridx.dx; dy = sys.gridy.dx
    return 0.1 * cfl * dx * dy / (v * (dx + dy))
end

function update_physical_fluxes!(flux, Q, sys)
    Fx = flux.Fx; Fy = flux.Fy
    nx = sys.gridx.nx; ny = sys.gridy.nx
    for j in 1:ny, i in 1:nx
        Fx[i, j, 1] = Q[i, j, 6]
        # ...
    end
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
    @. smooth.G₊ = Q.ρ + Q.ρ * (Q.u^2 + Q.v^2 + Q.w^2) + Q.P + 
                   Q.Bx^2 + Q.By^2 + Q.Bz^2 + α * Q.ρ * (Q.u + Q.v + Q.w)
    @. smooth.G₋ = Q.ρ + Q.ρ * (Q.u^2 + Q.v^2 + Q.w^2) + Q.P +
                   Q.Bx^2 + Q.By^2 + Q.Bz^2 - α * Q.ρ * (Q.u + Q.v + Q.w)
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
Periodic boundary conditions
"""
function boundary_conditions!(Q, sys, bctype::Periodic)
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

function plot_system(q, sys, titlename, filename)
    crx = sys.gridx.x[sys.gridx.cr_mesh]; cry = sys.gridy.x[sys.gridy.cr_mesh]

    q_transposed = q[crx, cry] |> transpose
    plt = Plots.contour(crx, cry, q_transposed, title=titlename, 
                        fill=true, linecolor=:plasma, levels=100, aspect_ratio=1.0)
    display(plt)
    # Plots.pdf(plt, filename)
end


function idealmhd(; γ=5/3, cfl=0.4, t_max=1.0)
    gridx = Weno.grid(size=256, min=0.0, max=2π)
    gridy = Weno.grid(size=256, min=0.0, max=2π)
    sys = SystemParameters2D(gridx, gridy, γ)
    rkpar = Weno.preallocate_rungekutta_parameters(gridx, gridy)
    wepar = Weno.preallocate_weno_parameters()
    state = preallocate_statevectors(sys)
    flux = preallocate_fluxes(sys)
    flxrec = preallocate_fluxreconstruction(sys)
    smooth = preallocate_smoothnessfunctions(sys)

    orszagtang!(state.Q, sys)
end

                    
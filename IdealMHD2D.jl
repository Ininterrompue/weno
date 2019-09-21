include("./System.jl")
include("./Weno.jl")
using Printf, LinearAlgebra
import Plots, BenchmarkTools
import Base.sign
Plots.pyplot()

abstract type BoundaryCondition end
struct Periodic <: BoundaryCondition end

"""
Primitive: (u, v, w, P)
Conserved: (ρ, ρu, ρv, ρw, Bx, By, Bz, E)
"""
struct StateVectors{T}
    Q_prim::Array{T, 3}   # real space primitive variables
    Q_cons::Array{T, 3}   # real space conserved, dynamic variables
    Az::Matrix{T}         # vector potential
    Q_proj::Array{T, 3}   # characteristic space
    Q_local::Matrix{T}    # local real space
    Az_local::Vector{T}   # local derivative of vector potential
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
    Az_x::Array{T, 3}     # x-derivative of Az, 1 → +, 2 → -
    Az_y::Array{T, 3}     # x-derivative of Az, 1 → +, 2 → -
    F_local::Matrix{T}    # local physical flux, real space
end

mutable struct FluxReconstruction{T}
    Q_avgprim::Array{T, 3}   # averaged nonconserved quantities
    Q_avgcons::Array{T, 3}   # averaged conserved quantities
    Lx::Matrix{T}            # left eigenvectors
    Ly::Matrix{T}
    Rx::Matrix{T}            # right eigenvectors
    Ry::Matrix{T}
end

struct SmoothnessFunctions{T}
    G₊::Matrix{T}
    G₋::Matrix{T}
end


function preallocate_statevectors(sys)
    nx = sys.gridx.nx; ny = sys.gridy.nx
    nprim = sys.nprim; ncons = sys.ncons
    Q_prim = zeros(nx, ny, nprim)
    Q_cons = zeros(nx, ny, ncons)
    Az = zeros(nx, ny)
    Q_proj = zeros(nx, ny, ncons)
    Q_local = zeros(6, ncons)
    Az_local = zeros(6)
    return StateVectors(Q_prim, Q_cons, Az, Q_proj, Q_local, Az_local)
end

function preallocate_fluxes(sys)
    nx = sys.gridx.nx; ny = sys.gridy.nx;
    ncons = sys.ncons
    Fx = zeros(nx, ny, ncons); Fy = zeros(nx, ny, ncons)
    Gx = zeros(nx, ny, ncons); Gy = zeros(nx, ny, ncons);
    Fx_hat = zeros(nx+1, ny+1, ncons); Fy_hat = zeros(nx+1, ny+1, ncons)
    Gx_hat = zeros(nx+1, ny+1, ncons); Gy_hat = zeros(nx+1, ny+1, ncons)
    Az_x = zeros(nx, ny, 2); Az_y = zeros(nx, ny, 2)
    F_local = zeros(6, ncons)
    return Fluxes(Fx, Fy, Gx, Gy, Fx_hat, Fy_hat, Gx_hat, Gy_hat, Az_x, Az_y, F_local)
end

function preallocate_fluxreconstruction(sys)
    nx = sys.gridx.nx; ny = sys.gridy.nx
    nprim = sys.nprim; ncons = sys.ncons
    Q_avgprim = zeros(nx+1, ny+1, nprim)
    Q_avgcons = zeros(nx+1, ny+1, ncons)
    evecLx = zeros(ncons, ncons); evecLy = zeros(ncons, ncons)
    evecRx = zeros(ncons, ncons); evecRy = zeros(ncons, ncons)
    return FluxReconstruction(Q_avgprim, Q_avgcons, evecLx, evecLy, evecRx, evecRy)
end

function preallocate_smoothnessfunctions(sys)
    nx = sys.gridx.nx; ny = sys.gridy.nx
    G₊ = zeros(nx, ny)
    G₋ = zeros(nx, ny)
    return SmoothnessFunctions(G₊, G₋)
end

mag2(x, y) = x^2 + y^2
mag2(x, y, z) = x^2 + y^2 + z^2

"""
It is important that sign is defined so that sign(0) = 1 to avoid singular eigenvectors. 
"""
sign(x::Real) = x >= 0 ? oneunit(x) : x/abs(x)

"""
Orszag-Tang vortex
(x, y) = [0, 2π] × [0, 2π], t_max = 2, 3, 4
"""
function orszagtang!(state, sys)
    γ = sys.γ; Az = state.Az
    Q_prim = state.Q_prim; Q_cons = state.Q_cons
    nx = sys.gridx.nx; ny = sys.gridy.nx

    for j in 1:ny, i in 1:nx
        x = sys.gridx.x[i]
        y = sys.gridy.x[j]
        Q_cons[i, j, 1] = γ^2       # ρ
        Q_prim[i, j, 4] = γ         # P
        Q_prim[i, j, 1] = -sin(y)   # u
        Q_prim[i, j, 2] = +sin(x)   # v
        Q_cons[i, j, 5] = -sin(y)   # Bx
        Q_cons[i, j, 6] = sin(2x)   # By
        Az[i, j] = 1/2 * cos(2x) + cos(y)
    end
end

function primitive_to_conserved!(state, sys)
    nx = sys.gridx.nx; ny = sys.gridy.nx; γ = sys.γ
    Q_prim = state.Q_prim; Q_cons = state.Q_cons
    for j in 1:ny, i in 1:nx
        Q_cons[i, j, 2]  = Q_cons[i, j, 1] * Q_prim[i, j, 1]   # ρu = ρ * u
        Q_cons[i, j, 3]  = Q_cons[i, j, 1] * Q_prim[i, j, 2]   # ρv = ρ * v
        Q_cons[i, j, 4]  = Q_cons[i, j, 1] * Q_prim[i, j, 3]   # ρw = ρ * w
        Q_cons[i, j, 8] = Q_prim[i, j, 4] / (γ-1) +            # E  = P / (γ-1) + 1/2 * ρU^2 + 1/2 * B^2
            1/2 * Q_cons[i, j, 1] * (mag2(Q_prim[i, j, 1], Q_prim[i, j, 2], Q_prim[i, j, 3])) +
            1/2 * mag2(Q_cons[i, j, 5], Q_cons[i, j, 6], Q_cons[i, j, 7])
    end
end

function conserved_to_primitive!(state, sys)
    nx = sys.gridx.nx; ny = sys.gridy.nx; γ = sys.γ
    Q_prim = state.Q_prim; Q_cons = state.Q_cons
    for j in 1:ny, i in 1:nx
        Q_prim[i, j, 1] = Q_cons[i, j, 2] / Q_cons[i, j, 1]   # u = ρu / ρ
        Q_prim[i, j, 2] = Q_cons[i, j, 3] / Q_cons[i, j, 1]   # v = ρv / ρ
        Q_prim[i, j, 3] = Q_cons[i, j, 4] / Q_cons[i, j, 1]   # w = ρw / ρ
        Q_prim[i, j, 4] = (γ-1) * ( Q_cons[i, j, 8] -         # P = (γ-1) * (E - 1/2 * ρU^2/ρ - 1/2 * B^2)
            1/2 * mag2(Q_cons[i, j, 2], Q_cons[i, j, 3], Q_cons[i, j, 4]) / Q_cons[i, j, 1] -
            1/2 * mag2(Q_cons[i, j, 5], Q_cons[i, j, 6], Q_cons[i, j, 7]) )
    end
end

function arithmetic_average!(i, j, state, flxrec, sys, dim)
    Q_prim = state.Q_prim; Q_cons = state.Q_cons
    Q_avgprim = flxrec.Q_avgprim; Q_avgcons = flxrec.Q_avgcons
    if dim == :X
        for k in 1:sys.nprim
            Q_avgprim[i, j, k] = 1/2 * (Q_prim[i, j, k] + Q_prim[i+1, j, k])
        end
        for k in 1:sys.ncons
            Q_avgcons[i, j, k] = 1/2 * (Q_cons[i, j, k] + Q_cons[i+1, j, k])
        end
    elseif dim == :Y
        for k in 1:sys.nprim
            Q_avgprim[i, j, k] = 1/2 * (Q_prim[i, j, k] + Q_prim[i, j+1, k])
        end
        for k in 1:sys.ncons
            Q_avgcons[i, j, k] = 1/2 * (Q_cons[i, j, k] + Q_cons[i, j+1, k])
        end    
    end
end

function slow_magnetosonic(ρ, P, Bx, By, Bz, γ, dim)
    a2 = γ*P/ρ; b2 = (Bx^2 + By^2 + Bz^2)/ρ
    if dim == :X 
        bn2 = Bx^2/ρ
    elseif dim == :Y 
        bn2 = By^2/ρ
    end
    return 1/sqrt(2) * sqrt(a2 + b2 - sqrt((a2 + b2)^2 - 4a2 * bn2))
end

function fast_magnetosonic(ρ, P, Bx, By, Bz, γ, dim)
    a2 = γ*P/ρ; b2 = (Bx^2 + By^2 + Bz^2)/ρ
    if dim == :X 
        bn2 = Bx^2/ρ
    elseif dim == :Y 
        bn2 = By^2/ρ
    end
    return 1/sqrt(2) * sqrt(a2 + b2 + sqrt((a2 + b2)^2 - 4a2 * bn2))
end

function max_eigval(state, sys, dim)
    Q_prim = state.Q_prim; Q_cons = state.Q_cons; γ = sys.γ
    return @views @. $maximum( abs(Q_prim[:, :, 1]) +
        fast_magnetosonic(Q_cons[:, :, 1], Q_prim[:, :, 4], 
                          Q_cons[:, :, 5], Q_cons[:, :, 6], 
                          Q_cons[:, :, 7], γ, dim) )
end 

function CFL_condition(cfl, state, sys, wepar)
    dx = sys.gridx.dx; dy = sys.gridy.dx
    vx = max_eigval(state, sys, :X)
    vy = max_eigval(state, sys, :Y)
    wepar.ev = max(vx, vy)
    return 0.05 * cfl * dx*dy / (vx*dy + vy*dx)
end

function update_physical_fluxes!(flux, state, sys)
    Fx = flux.Fx; nx = sys.gridx.nx
    Fy = flux.Fy; ny = sys.gridy.nx
    Q_prim = state.Q_prim; Q_cons = state.Q_cons
    for j in 1:ny, i in 1:nx
        ρ  = Q_cons[i, j, 1]; ρu = Q_cons[i, j, 2]
        ρv = Q_cons[i, j, 3]; ρw = Q_cons[i, j, 4]
        Bx = Q_cons[i, j, 5]; By = Q_cons[i, j, 6]
        Bz = Q_cons[i, j, 7]; E  = Q_cons[i, j, 8]
        u = Q_prim[i, j, 1]; v = Q_prim[i, j, 2]
        w = Q_prim[i, j, 3]; P = Q_prim[i, j, 4]
        P_tot = P + 1/2 * mag2(Bx, By, Bz)

        Fx[i, j, 1] = ρu
        Fx[i, j, 2] = ρ * u^2 + P_tot - Bx^2
        Fx[i, j, 3] = ρ * u*v - Bx * By
        Fx[i, j, 4] = ρ * u*w - Bx * Bz
      # Fx[i, j, 5] = 0
        Fx[i, j, 6] = u * By - v * Bx
        Fx[i, j, 7] = u * Bz - w * Bx
        Fx[i, j, 8] = u * (E + P_tot) - Bx * (u*Bx + v*By + w*Bz)

        Fy[i, j, 1] = ρv
        Fy[i, j, 2] = ρ * v*u - By * Bx
        Fy[i, j, 3] = ρ * v^2 + P_tot - By^2
        Fy[i, j, 4] = ρ * v*w - By * Bz
        Fy[i, j, 5] = v * Bx - u * By
      # Fy[i, j, 6] = 0
        Fy[i, j, 7] = v * Bz - w * By
        Fy[i, j, 8] = v * (E + P_tot) - By * (u*Bx + v*By + w*Bz)
    end
end

function update_xeigenvectors!(i, j, state, flxrec, sys)
    Rx = flxrec.Rx; Lx = flxrec.Lx; γ = sys.γ
    Q_avgprim = flxrec.Q_avgprim
    Q_avgcons = flxrec.Q_avgcons

    arithmetic_average!(i, j, state, flxrec, sys, :X)

    ρ  = Q_avgcons[i, j, 1]; P  = Q_avgprim[i, j, 4]; E  = Q_avgcons[i, j, 8]
    u  = Q_avgprim[i, j, 1]; v  = Q_avgprim[i, j, 2]; w  = Q_avgprim[i, j, 3]
    Bx = Q_avgcons[i, j, 5]; By = Q_avgcons[i, j, 6]; Bz = Q_avgcons[i, j, 7]
    a = sqrt(γ*P/ρ)   # sound speed
    cs = slow_magnetosonic(ρ, P, Bx, By, Bz, γ, :X)
    cf = fast_magnetosonic(ρ, P, Bx, By, Bz, γ, :X)

    if By^2 + Bz^2 > 1e-12 * mag2(Bx, By, Bz)
        βy = By / sqrt(mag2(By, Bz))
        βz = Bz / sqrt(mag2(By, Bz))
    else
        βy = 1/sqrt(2)
        βz = 1/sqrt(2)
    end
    if By^2 + Bz^2 > 1e-12 * mag2(Bx, By, Bz) || abs(γ*P - Bx^2) > 1e-12 * γ*P
        αf = a^2 - cs^2 < 0.0 ? 0.0 : sqrt(a^2 - cs^2) / sqrt(cf^2 - cs^2)
        αs = cf^2 - a^2 < 0.0 ? 0.0 : sqrt(cf^2 - a^2) / sqrt(cf^2 - cs^2)
    else
        αf = 1.0
        αs = 1.0
    end
    γ1 = (γ-1)/2
    γ2 = (γ-2)/(γ-1)
    τ  = (γ-1)/a^2
    Γf = αf * cf * u - αs * cs * sign(Bx) * (βy * v + βz * w)
    Γa = sign(Bx) * (βz * v - βy * w)
    Γs = αs * cs * u + αf * cf * sign(Bx) * (βy * v + βz * w)
    oneover2a2 = 1/2a^2
    
    Rx[1, 1] = αf
    Rx[2, 1] = αf * (u - cf)
    Rx[3, 1] = αf * v + cs * αs * βy * sign(Bx)
    Rx[4, 1] = αf * w + cs * αs * βz * sign(Bx)
    Rx[6, 1] = a * αs * βy / sqrt(ρ)
    Rx[7, 1] = a * αs * βz / sqrt(ρ)
    Rx[8, 1] = αf * (1/2 * mag2(u, v, w) + cf^2 - γ2 * a^2) - Γf

    Rx[3, 2] = -βz * sign(Bx)
    Rx[4, 2] = +βy * sign(Bx)
    Rx[6, 2] = -βz / sqrt(ρ)
    Rx[7, 2] = +βy / sqrt(ρ)
    Rx[8, 2] = -Γa

    Rx[1, 3] = αs
    Rx[2, 3] = αs * (u - cs)
    Rx[3, 3] = αs * v - cf * αf * βy * sign(Bx)
    Rx[4, 3] = αs * w - cf * αf * βz * sign(Bx)
    Rx[6, 3] = -a * αf * βy / sqrt(ρ)
    Rx[7, 3] = -a * αf * βz / sqrt(ρ)
    Rx[8, 3] = αs * (1/2 * mag2(u, v, w) + cs^2 - γ2 * a^2) - Γs

    Rx[1, 4] = 1
    Rx[2, 4] = u
    Rx[3, 4] = v
    Rx[4, 4] = w
    Rx[8, 4] = 1/2 * mag2(u, v, w)

    Rx[5, 5] = 1
    Rx[8, 5] = Bx

    Rx[1, 6] = αs
    Rx[2, 6] = αs * (u + cs)
    Rx[3, 6] = αs * v + cf * αf * βy * sign(Bx)
    Rx[4, 6] = αs * w + cf * αf * βz * sign(Bx)
    Rx[6, 6] = -a * αf * βy / sqrt(ρ)
    Rx[7, 6] = -a * αf * βz / sqrt(ρ)
    Rx[8, 6] = αs * (1/2 * mag2(u, v, w) + cs^2 - γ2 * a^2) + Γs

    Rx[3, 7] = -βz * sign(Bx)
    Rx[4, 7] = +βy * sign(Bx)
    Rx[6, 7] = +βz / sqrt(ρ)
    Rx[7, 7] = -βy / sqrt(ρ)
    Rx[8, 7] = -Γa

    Rx[1, 8] = αf
    Rx[2, 8] = αf * (u + cf)
    Rx[3, 8] = αf * v - cs * αs * βy * sign(Bx)
    Rx[4, 8] = αf * w - cs * αs * βz * sign(Bx)
    Rx[6, 8] = a * αs * βy / sqrt(ρ)
    Rx[7, 8] = a * αs * βz / sqrt(ρ)
    Rx[8, 8] = αf * (1/2 * mag2(u, v, w) + cf^2 - γ2 * a^2) + Γf

    # flxrec.Lx = inv(Rx)

    Lx[1, 1] = oneover2a2 * (γ1 * αf * mag2(u, v, w) + Γf)
    Lx[1, 2] = oneover2a2 * ((1-γ) * αf * u - αf * cf)
    Lx[1, 3] = oneover2a2 * ((1-γ) * αf * v + cs * αs * βy * sign(Bx))
    Lx[1, 4] = oneover2a2 * ((1-γ) * αf * w + cs * αs * βz * sign(Bx))
    Lx[1, 5] = oneover2a2 * ((1-γ) * αf * Bx)
    Lx[1, 6] = oneover2a2 * ((1-γ) * αf * By + sqrt(ρ) * a * αs * βy) # +
    Lx[1, 7] = oneover2a2 * ((1-γ) * αf * Bz - sqrt(ρ) * a * αs * βz)
    Lx[1, 8] = oneover2a2 * ((γ-1) * αf)

    Lx[2, 1] = 1/2 * Γa
    Lx[2, 3] = 1/2 * -βz * sign(Bx)
    Lx[2, 4] = 1/2 * +βy * sign(Bx)
    Lx[2, 6] = 1/2 * -sqrt(ρ) * βz
    Lx[2, 7] = 1/2 * +sqrt(ρ) * βy

    Lx[3, 1] = oneover2a2 * (γ1 * αs * mag2(u, v, w) + Γs)
    Lx[3, 2] = oneover2a2 * ((1-γ) * αs * u - αs * cs)
    Lx[3, 3] = oneover2a2 * ((1-γ) * αs * v - cf * αf * βy * sign(Bx))
    Lx[3, 4] = oneover2a2 * ((1-γ) * αs * w - cf * αf * βz * sign(Bx))
    Lx[3, 5] = oneover2a2 * ((1-γ) * αs * Bx)
    Lx[3, 6] = oneover2a2 * ((1-γ) * αs * By - sqrt(ρ) * a * αf * βy)
    Lx[3, 7] = oneover2a2 * ((1-γ) * αs * Bz - sqrt(ρ) * a * αf * βz)
    Lx[3, 8] = oneover2a2 * ((γ-1) * αs)

    Lx[4, 1] = 1 - 1/2 * τ * mag2(u, v, w)
    Lx[4, 2] = τ * u
    Lx[4, 3] = τ * v
    Lx[4, 4] = τ * w
    Lx[4, 5] = τ * Bx
    Lx[4, 6] = τ * By
    Lx[4, 7] = τ * Bz
    Lx[4, 8] = -τ

    Lx[5, 5] = 1.0 

    Lx[6, 1] = oneover2a2 * (γ1 * αs * mag2(u, v, w) - Γs)
    Lx[6, 2] = oneover2a2 * ((1-γ) * αs * u + αs * cs)
    Lx[6, 3] = oneover2a2 * ((1-γ) * αs * v + cf * αf * βy * sign(Bx))
    Lx[6, 4] = oneover2a2 * ((1-γ) * αs * w + cf * αf * βz * sign(Bx))
    Lx[6, 5] = oneover2a2 * ((1-γ) * αs * Bx)
    Lx[6, 6] = oneover2a2 * ((1-γ) * αs * By - sqrt(ρ) * a * αf * βy)
    Lx[6, 7] = oneover2a2 * ((1-γ) * αs * Bz - sqrt(ρ) * a * αf * βz)
    Lx[6, 8] = oneover2a2 * ((γ-1) * αs)

    Lx[7, 1] = 1/2 * Γa
    Lx[7, 3] = 1/2 * -βz * sign(Bx)
    Lx[7, 4] = 1/2 * +βy * sign(Bx)
    Lx[7, 6] = 1/2 * +sqrt(ρ) * βz
    Lx[7, 7] = 1/2 * -sqrt(ρ) * βy

    Lx[8, 1] = oneover2a2 * (γ1 * αf * mag2(u, v, w) - Γf)
    Lx[8, 2] = oneover2a2 * ((1-γ) * αf * u + αf * cf)
    Lx[8, 3] = oneover2a2 * ((1-γ) * αf * v - cs * αs * βy * sign(Bx))
    Lx[8, 4] = oneover2a2 * ((1-γ) * αf * w - cs * αs * βz * sign(Bx))
    Lx[8, 5] = oneover2a2 * ((1-γ) * αf * Bx)
    Lx[8, 6] = oneover2a2 * ((1-γ) * αf * By + sqrt(ρ) * a * αs * βy) # +
    Lx[8, 7] = oneover2a2 * ((1-γ) * αf * Bz - sqrt(ρ) * a * αs * βz)
    Lx[8, 8] = oneover2a2 * ((γ-1) * αf)
end

function update_yeigenvectors!(i, j, state, flxrec, sys)
    Ry = flxrec.Ry; Ly = flxrec.Ly; γ = sys.γ
    Q_avgprim = flxrec.Q_avgprim
    Q_avgcons = flxrec.Q_avgcons

    arithmetic_average!(i, j, state, flxrec, sys, :Y)

    ρ  = Q_avgcons[i, j, 1]; P  = Q_avgprim[i, j, 4]; E  = Q_avgcons[i, j, 8]
    u  = Q_avgprim[i, j, 1]; v  = Q_avgprim[i, j, 2]; w  = Q_avgprim[i, j, 3]
    Bx = Q_avgcons[i, j, 5]; By = Q_avgcons[i, j, 6]; Bz = Q_avgcons[i, j, 7]
    a = sqrt(γ*P/ρ)   # sound speed
    cs = slow_magnetosonic(ρ, P, Bx, By, Bz, γ, :Y)
    cf = fast_magnetosonic(ρ, P, Bx, By, Bz, γ, :Y)

    if Bx^2 + Bz^2 > 1e-12 * mag2(Bx, By, Bz)
        βx = Bx / sqrt(mag2(Bx, Bz))
        βz = Bz / sqrt(mag2(Bx, Bz))
    else
        βx = 1/sqrt(2)
        βz = 1/sqrt(2)
    end
    if Bx^2 + Bz^2 > 1e-12 * mag2(Bx, By, Bz) || abs(γ*P - By^2) > 1e-12 * γ*P
        αf = a^2 - cs^2 < 0.0 ? 0.0 : sqrt(a^2 - cs^2) / sqrt(cf^2 - cs^2)
        αs = cf^2 - a^2 < 0.0 ? 0.0 : sqrt(cf^2 - a^2) / sqrt(cf^2 - cs^2)
    else
        αf = 1.0
        αs = 1.0
    end
    γ1 = (γ-1)/2
    γ2 = (γ-2)/(γ-1)
    τ  = (γ-1)/a^2
    Γf = αf * cf * v - αs * cs * sign(By) * (βx * u + βz * w)
    Γa = sign(By) * (βz * u - βx * w)
    Γs = αs * cs * v + αf * cf * sign(By) * (βx * v + βz * w)
    oneover2a2 = 1/2a^2
    
    Ry[1, 1] = αf
    Ry[2, 1] = αf * (v - cf)
    Ry[3, 1] = αf * u + cs * αs * βx * sign(By)
    Ry[4, 1] = αf * w + cs * αs * βz * sign(By)
    Ry[6, 1] = a * αs * βx / sqrt(ρ)
    Ry[7, 1] = a * αs * βz / sqrt(ρ)
    Ry[8, 1] = αf * (1/2 * mag2(u, v, w) + cf^2 - γ2 * a^2) - Γf

    Ry[3, 3] = -βz * sign(By)
    Ry[4, 3] = +βx * sign(By)
    Ry[6, 3] = -βz / sqrt(ρ)
    Ry[7, 3] = +βx / sqrt(ρ)
    Ry[8, 3] = -Γa

    Ry[1, 2] = αs
    Ry[2, 2] = αs * (v - cs)
    Ry[3, 2] = αs * u - cf * αf * βx * sign(By)
    Ry[4, 2] = αs * w - cf * αf * βz * sign(By)
    Ry[6, 2] = -a * αf * βx / sqrt(ρ)
    Ry[7, 2] = -a * αf * βz / sqrt(ρ)
    Ry[8, 2] = αs * (1/2 * mag2(u, v, w) + cs^2 - γ2 * a^2) - Γs

    Ry[1, 4] = 1
    Ry[2, 4] = v
    Ry[3, 4] = u
    Ry[4, 4] = w
    Ry[8, 4] = 1/2 * mag2(u, v, w)

    Ry[5, 6] = 1
    Ry[8, 6] = By

    Ry[1, 5] = αs
    Ry[2, 5] = αs * (v + cs)
    Ry[3, 5] = αs * u + cf * αf * βx * sign(By)
    Ry[4, 5] = αs * w + cf * αf * βz * sign(By)
    Ry[6, 5] = -a * αf * βx / sqrt(ρ)
    Ry[7, 5] = -a * αf * βz / sqrt(ρ)
    Ry[8, 5] = αs * (1/2 * mag2(u, v, w) + cs^2 - γ2 * a^2) + Γs

    Ry[3, 7] = -βz * sign(By)
    Ry[4, 7] = +βx * sign(By)
    Ry[6, 7] = +βz / sqrt(ρ)
    Ry[7, 7] = -βx / sqrt(ρ)
    Ry[8, 7] = -Γa

    Ry[1, 8] = αf
    Ry[2, 8] = αf * (v + cf)
    Ry[3, 8] = αf * u - cs * αs * βx * sign(By)
    Ry[4, 8] = αf * w - cs * αs * βz * sign(By)
    Ry[6, 8] = a * αs * βx / sqrt(ρ)
    Ry[7, 8] = a * αs * βz / sqrt(ρ)
    Ry[8, 8] = αf * (1/2 * mag2(u, v, w) + cf^2 - γ2 * a^2) + Γf

    # flxrec.Ly = inv(Ry)
    # return 

    # Problems with Ly, not Lx
    Ly[1, 1] = oneover2a2 * (γ1 * αf * mag2(u, v, w) + Γf)
    Ly[1, 2] = oneover2a2 * ((1-γ) * αf * v - αf * cf)
    Ly[1, 3] = oneover2a2 * ((1-γ) * αf * u + cs * αs * βx * sign(By))
    Ly[1, 4] = oneover2a2 * ((1-γ) * αf * w + cs * αs * βz * sign(By))
    Ly[1, 5] = oneover2a2 * ((1-γ) * αf * By)
    Ly[1, 6] = oneover2a2 * ((1-γ) * αf * By + sqrt(ρ) * a * αs * βx) # +
    Ly[1, 7] = oneover2a2 * ((1-γ) * αf * Bz - sqrt(ρ) * a * αs * βz)
    Ly[1, 8] = oneover2a2 * ((γ-1) * αf)

    Ly[2, 1] = 1/2 * Γa
    Ly[2, 3] = 1/2 * -βz * sign(By)
    Ly[2, 4] = 1/2 * +βx * sign(By)
    Ly[2, 6] = 1/2 * -sqrt(ρ) * βz
    Ly[2, 7] = 1/2 * +sqrt(ρ) * βx

    Ly[3, 1] = oneover2a2 * (γ1 * αs * mag2(u, v, w) + Γs)
    Ly[3, 2] = oneover2a2 * ((1-γ) * αs * v - αs * cs)
    Ly[3, 3] = oneover2a2 * ((1-γ) * αs * u - cf * αf * βx * sign(By))
    Ly[3, 4] = oneover2a2 * ((1-γ) * αs * w - cf * αf * βz * sign(By))
    Ly[3, 5] = oneover2a2 * ((1-γ) * αs * By)
    Ly[3, 6] = oneover2a2 * ((1-γ) * αs * By - sqrt(ρ) * a * αf * βx)
    Ly[3, 7] = oneover2a2 * ((1-γ) * αs * Bz - sqrt(ρ) * a * αf * βz)
    Ly[3, 8] = oneover2a2 * ((γ-1) * αs)

    Ly[4, 1] = 1 - 1/2 * τ * mag2(u, v, w)
    Ly[4, 2] = τ * v
    Ly[4, 3] = τ * u
    Ly[4, 4] = τ * w
    Ly[4, 5] = τ * By
    Ly[4, 6] = τ * Bx
    Ly[4, 7] = τ * Bz
    Ly[4, 8] = -τ

    Ly[5, 5] = 1.0 

    Ly[6, 1] = oneover2a2 * (γ1 * αs * mag2(u, v, w) - Γs)
    Ly[6, 2] = oneover2a2 * ((1-γ) * αs * v + αs * cs)
    Ly[6, 3] = oneover2a2 * ((1-γ) * αs * u + cf * αf * βx * sign(By))
    Ly[6, 4] = oneover2a2 * ((1-γ) * αs * w + cf * αf * βz * sign(By))
    Ly[6, 5] = oneover2a2 * ((1-γ) * αs * By)
    Ly[6, 6] = oneover2a2 * ((1-γ) * αs * By - sqrt(ρ) * a * αf * βx)
    Ly[6, 7] = oneover2a2 * ((1-γ) * αs * Bz - sqrt(ρ) * a * αf * βz)
    Ly[6, 8] = oneover2a2 * ((γ-1) * αs)

    Ly[7, 1] = 1/2 * Γa
    Ly[7, 3] = 1/2 * -βz * sign(By)
    Ly[7, 4] = 1/2 * +βx * sign(By)
    Ly[7, 6] = 1/2 * +sqrt(ρ) * βz
    Ly[7, 7] = 1/2 * -sqrt(ρ) * βx

    Ly[8, 1] = oneover2a2 * (γ1 * αf * mag2(u, v, w) - Γf)
    Ly[8, 2] = oneover2a2 * ((1-γ) * αf * v + αf * cf)
    Ly[8, 3] = oneover2a2 * ((1-γ) * αf * u - cs * αs * βx * sign(By))
    Ly[8, 4] = oneover2a2 * ((1-γ) * αf * w - cs * αs * βz * sign(By))
    Ly[8, 5] = oneover2a2 * ((1-γ) * αf * By)
    Ly[8, 6] = oneover2a2 * ((1-γ) * αf * By + sqrt(ρ) * a * αs * βx) # +
    Ly[8, 7] = oneover2a2 * ((1-γ) * αf * Bz - sqrt(ρ) * a * αs * βz)
    Ly[8, 8] = oneover2a2 * ((γ-1) * αf)
end

function update_local!(i, j, Q, F, Q_local, F_local, sys, dim)
    if dim == :X
        for n in 1:sys.ncons, k in 1:6
            Q_local[k, n] = Q[i-3+k, j, n]
            F_local[k, n] = F[i-3+k, j, n]
        end
    elseif dim == :Y
        for n in 1:sys.ncons, k in 1:6
            Q_local[k, n] = Q[i, j-3+k, n]
            F_local[k, n] = F[i, j-3+k, n]
        end
    end
end

function update_smoothnessfunctions!(smooth, state, sys, α)
    nx = sys.gridx.nx; ny = sys.gridy.nx
    Q_prim = state.Q_prim; Q_cons = state.Q_cons
    for j in 1:ny, i in 1:nx
        ρ  = Q_cons[i, j, 1]
        ρu = Q_cons[i, j, 2]
        ρv = Q_cons[i, j, 3]
        ρw = Q_cons[i, j, 4]
        E  = Q_cons[i, j, 8]

        smooth.G₊[i] = ρ + E + α * (ρu + ρv + ρw)
        smooth.G₋[i] = ρ + E - α * (ρu + ρv + ρw)
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

function project_to_localspace!(i, j, state, flux, flxrec, sys, dim)
    Q_cons = state.Q_cons; Q_proj = state.Q_proj
    Fx = flux.Fx; Gx = flux.Gx; Lx = flxrec.Lx
    Fy = flux.Fy; Gy = flux.Gy; Ly = flxrec.Ly

    if dim == :X
        for n in 1:sys.ncons, k in i-2:i+3
            # Q_proj[k, j, n] = @views Lx[n, :] ⋅ Q_cons[k, j, :]
            # Gx[k, j, n] = @views Lx[n, :] ⋅ Fx[k, j, :]
            Q_proj[k, j, n] = 0.0
            Gx[k, j, n] = 0.0
            for m in 1:sys.ncons
                Q_proj[k, j, n] += Lx[n, m] * Q_cons[k, j, m]
                Gx[k, j, n] += Lx[n, m] * Fx[k, j, m]
            end
        end
    elseif dim == :Y
        for n in 1:sys.ncons, k in j-2:j+3
            # Q_proj[i, k, n] = @views Ly[n, :] ⋅ Q_cons[i, k, :]
            # Gy[i, k, n] = @views Ly[n, :] ⋅ Fy[i, k, :]
            Q_proj[i, k, n] = 0.0
            Gy[i, k, n] = 0.0
            for m in 1:sys.ncons
                Q_proj[i, k, n] += Ly[n, m] * Q_cons[i, k, m]
                Gy[i, k, n] += Ly[n, m] * Fy[i, k, m]
            end
        end
    end
end

function project_to_realspace!(i, j, flux, flxrec, sys, dim)
    Fx_hat = flux.Fx_hat; Gx_hat = flux.Gx_hat; Rx = flxrec.Rx
    Fy_hat = flux.Fy_hat; Gy_hat = flux.Gy_hat; Ry = flxrec.Ry

    if dim == :X
        for n in 1:sys.ncons
            # Fx_hat[i, j, n] = @views Rx[n, :] ⋅ Gx_hat[i, j, :]
            Fx_hat[i, j, n] = 0.0
            for m in 1:sys.ncons
                Fx_hat[i, j, n] += Rx[n, m] * Gx_hat[i, j, m]
            end
        end
    elseif dim == :Y
        for n in 1:sys.ncons
            # Fy_hat[i, j, n] = @views Ry[n, :] ⋅ Gy_hat[i, j, :]
            Fy_hat[i, j, n] = 0.0
            for m in 1:sys.ncons
                Fy_hat[i, j, n] += Ry[n, m] * Gy_hat[i, j, m]
            end
        end
    end
end

function time_evolution!(state, flux, sys, dt, rkpar)
    for n in 1:sys.ncons, j in sys.gridy.cr_mesh, i in sys.gridx.cr_mesh
        rkpar.op[i, j, n] = Weno.weno_scheme(
            flux.Fx_hat[i, j, n], flux.Fx_hat[i-1, j, n],
            flux.Fy_hat[i, j, n], flux.Fy_hat[i, j-1, n], 
            sys.gridx, sys.gridy, rkpar)
    end
    Weno.runge_kutta!(state.Q_cons, dt, rkpar)
end

"""
Updates the local derivative of the vector potential to control oscillations, 
as the derivative of Az produces a physical quantity.
"""
function update_local_Az_derivatives!(i, j, state, sys, wepar, dim)
    Az_local = state.Az_local; Az = state.Az
    dx = sys.gridx.dx; dy = sys.gridy.dx
    if dim == :X
        for k in 1:6
            Az_local[k] = 1/dx * (Az[i+k-3, j] - Az[i+k-4, j])
        end
    elseif dim == :Y
        for k in 1:6
            Az_local[k] = 1/dy * (Az[i, j+k-3] - Az[i, j+k-4])
        end
    end
    @. wepar.fp = state.Az_local
    @. wepar.fm = state.Az_local
end

function update_numerical_Az_derivatives!(i, j, state, flux, wepar, dim)
    if dim == :X
        flux.Az_x[i, j, 1] = Weno.fhatm(wepar, false)
        flux.Az_x[i, j, 2] = Weno.fhatp(wepar, false)
    elseif dim == :Y
        flux.Az_y[i, j, 1] = Weno.fhatm(wepar, false)
        flux.Az_y[i, j, 2] = Weno.fhatp(wepar, false)
    end
end

function time_evolution_Az!(state, flux, sys, dt, rApar)
    crx = sys.gridx.cr_mesh; cry = sys.gridy.cr_mesh
    Q_prim = state.Q_prim
    Az_x = flux.Az_x; Az_y = flux.Az_y
    αx = maximum(@view(Q_prim[:, :, 1]))
    αy = maximum(@view(Q_prim[:, :, 2]))
    for j in cry, i in crx
        rApar.op[i, j] = -Q_prim[i, j, 1]/2 * (Az_x[i, j, 1] + Az_x[i, j, 2]) + 
                         -Q_prim[i, j, 2]/2 * (Az_y[i, j, 1] + Az_y[i, j, 2]) +
                         αx/2 * (Az_x[i, j, 1] - Az_x[i, j, 2]) + 
                         αy/2 * (Az_y[i, j, 1] - Az_y[i, j, 2])
    end
    Weno.runge_kutta!(state.Az, dt, rApar)
end

function correct_magneticfield!(state, sys)
    Q_cons = state.Q_cons; Az = state.Az
    crx = sys.gridx.cr_mesh; cry = sys.gridy.cr_mesh
    dx = sys.gridx.dx; dy = sys.gridy.dx

    for j in cry, i in crx
        Q_cons[i, j, 5] = 1/dy * (+1/60 * Az[i, j+3] - 3/20 * Az[i, j+2] + 3/4  * Az[i, j+1] - 
                                  +3/4  * Az[i, j-1] + 3/20 * Az[i, j-2] - 1/60 * Az[i, j-3])
        Q_cons[i, j, 6] = 1/dx * (-1/60 * Az[i+3, j] + 3/20 * Az[i+2, j] - 3/4  * Az[i+1, j] +
                                  +3/4  * Az[i-1, j] - 3/20 * Az[i-2, j] + 1/60 * Az[i-3, j])
    end
end

function boundary_conditions_conserved!(state, sys, bctype::Periodic)
    Q_cons = state.Q_cons; ncons = sys.ncons
    nx = sys.gridx.nx; ny = sys.gridy.nx
    for n in 1:ncons, i in 1:nx
        Q_cons[i, end-0, n] = Q_cons[i, 6, n]
        Q_cons[i, end-1, n] = Q_cons[i, 5, n]
        Q_cons[i, end-2, n] = Q_cons[i, 4, n]
        Q_cons[i, 3, n] = Q_cons[i, end-3, n]
        Q_cons[i, 2, n] = Q_cons[i, end-4, n]
        Q_cons[i, 1, n] = Q_cons[i, end-5, n]
    end
    for n in 1:ncons, j in 1:ny
        Q_cons[end-0, j, n] = Q_cons[6, j, n]
        Q_cons[end-1, j, n] = Q_cons[5, j, n]
        Q_cons[end-2, j, n] = Q_cons[4, j, n]
        Q_cons[3, j, n] = Q_cons[end-3, j, n]
        Q_cons[2, j, n] = Q_cons[end-4, j, n]
        Q_cons[1, j, n] = Q_cons[end-5, j, n]
    end
end

function boundary_conditions_primitive!(state, sys, bctype::Periodic)
    Q_prim = state.Q_prim; nprim = sys.nprim
    nx = sys.gridx.nx; ny = sys.gridy.nx
    for n in 1:nprim, i in 1:nx
        Q_prim[i, end-0, n] = Q_prim[i, 6, n]
        Q_prim[i, end-1, n] = Q_prim[i, 5, n]
        Q_prim[i, end-2, n] = Q_prim[i, 4, n]
        Q_prim[i, 3, n] = Q_prim[i, end-3, n]
        Q_prim[i, 2, n] = Q_prim[i, end-4, n]
        Q_prim[i, 1, n] = Q_prim[i, end-5, n]
    end
    for n in 1:nprim, j in 1:ny
        Q_prim[end-0, j, n] = Q_prim[6, j, n]
        Q_prim[end-1, j, n] = Q_prim[5, j, n]
        Q_prim[end-2, j, n] = Q_prim[4, j, n]
        Q_prim[3, j, n] = Q_prim[end-3, j, n]
        Q_prim[2, j, n] = Q_prim[end-4, j, n]
        Q_prim[1, j, n] = Q_prim[end-5, j, n]
    end
end

function boundary_conditions_Az!(state, sys, bctype::Periodic)
    Az = state.Az
    nx = sys.gridx.nx; ny = sys.gridy.nx
    for i in 1:nx
        Az[i, end-0] = Az[i, 6]
        Az[i, end-1] = Az[i, 5]
        Az[i, end-2] = Az[i, 4]
        Az[i, 3] = Az[i, end-3]
        Az[i, 2] = Az[i, end-4]
        Az[i, 1] = Az[i, end-5]
    end
    for j in 1:ny
        Az[end-0, j] = Az[6, j]
        Az[end-1, j] = Az[5, j]
        Az[end-2, j] = Az[4, j]
        Az[3, j] = Az[end-3, j]
        Az[2, j] = Az[end-4, j]
        Az[1, j] = Az[end-5, j]
    end
end

function calculate_temperature(state, sys)
    nx = sys.gridx.nx; ny = sys.gridy.nx
    T = zeros(nx, ny)
    for j in 1:ny, i in 1:nx
        T[i, j] = state.Q_prim[i, j, 4] / 2state.Q_cons[i, j, 1]
    end
    return T
end

function calculate_divergence(state, sys)
    nx = sys.gridx.nx; ny = sys.gridy.nx
    dx = sys.gridx.dx; dy = sys.gridy.dx
    crx = sys.gridx.cr_mesh; cry = sys.gridy.cr_mesh
    Q_cons = state.Q_cons
    divB = zeros(nx, ny)
    for j in cry, i in crx
        divB[i, j] = 1/dx * (1/60 * Q_cons[i+3, j, 5] - 3/20 * Q_cons[i+2, j, 5] + 
                             3/4  * Q_cons[i+1, j, 5] - 3/4  * Q_cons[i-1, j, 5] +
                             3/20 * Q_cons[i-2, j, 5] - 1/60 * Q_cons[i-3, j, 5]) +
                     1/dy * (1/60 * Q_cons[i, j+3, 6] - 3/20 * Q_cons[i, j+2, 6] + 
                             3/4  * Q_cons[i, j+1, 6] - 3/4  * Q_cons[i, j-1, 6] + 
                             3/20 * Q_cons[i, j-2, 6] - 1/60 * Q_cons[i, j-3, 6])
    end
    return divB
end

function plot_system(q, sys, titlename, filename)
    crx = sys.gridx.x[sys.gridx.cr_mesh]; cry = sys.gridy.x[sys.gridy.cr_mesh]
    q_transposed = q[sys.gridx.cr_mesh, sys.gridy.cr_mesh] |> transpose
    plt = Plots.contour(crx, cry, q_transposed, title=titlename, 
                        fill=false, linecolor=:plasma, levels=15, aspect_ratio=1.0)
    display(plt)
    # Plots.pdf(plt, filename)
end


function idealmhd(; γ=5/3, cfl=0.4, t_max=0.0)
    gridx = grid(size=64, min=0.0, max=2π)
    gridy = grid(size=64, min=0.0, max=2π)
    sys = SystemParameters2D(gridx, gridy, 4, 8, γ)
    rApar = Weno.preallocate_rungekutta_parameters_2D(gridx, gridy)
    rkpar = Weno.preallocate_rungekutta_parameters_2D(gridx, gridy, sys)
    wepar = Weno.preallocate_weno_parameters()
    state = preallocate_statevectors(sys)
    flux = preallocate_fluxes(sys)
    flxrec = preallocate_fluxreconstruction(sys)
    smooth = preallocate_smoothnessfunctions(sys)

    orszagtang!(state, sys)
    bctype = Periodic()

    primitive_to_conserved!(state, sys)
    boundary_conditions_primitive!(state, sys, bctype)
    boundary_conditions_conserved!(state, sys, bctype)
    boundary_conditions_Az!(state, sys, bctype)

    t = 0.0; counter = 0; t0 = time()
    q = state.Q_local; f = flux.F_local
    t_array = [1.0, 2.0, 3.0, 4.0]; t_counter = 1
    while t < t_max
        update_physical_fluxes!(flux, state, sys)
        dt = CFL_condition(cfl, state, sys, wepar)
        t += dt
        
        # Component-wise reconstruction
        # for j in gridy.cr_cell, i in gridx.cr_cell
        #     update_local!(i, j, state.Q_cons, flux.Fx, q, f, sys, :X)
        #     Weno.update_numerical_fluxes!(i, j, flux.Fx_hat, q, f, sys, wepar, false)
        # end
        # for j in gridy.cr_cell, i in gridx.cr_cell
        #     update_local!(i, j, state.Q_cons, flux.Fy, q, f, sys, :Y)
        #     Weno.update_numerical_fluxes!(i, j, flux.Fy_hat, q, f, sys, wepar, false)
        # end

        # Characteristic-wise reconstruction
        # for j in gridy.cr_cell, i in gridx.cr_cell
        #     update_xeigenvectors!(i, j, state, flxrec, sys)
        #     project_to_localspace!(i, j, state, flux, flxrec, sys, :X)
        #     update_local!(i, j, state.Q_proj, flux.Gx, q, f, sys, :X)
        #     Weno.update_numerical_fluxes!(i, j, flux.Gx_hat, q, f, sys, wepar, false)
        #     project_to_realspace!(i, j, flux, flxrec, sys, :X)
        # end
        # for j in gridy.cr_cell, i in gridx.cr_cell
        #     update_yeigenvectors!(i, j, state, flxrec, sys)
        #     project_to_localspace!(i, j, state, flux, flxrec, sys, :Y)
        #     update_local!(i, j, state.Q_proj, flux.Gy, q, f, sys, :Y)
        #     Weno.update_numerical_fluxes!(i, j, flux.Gy_hat, q, f, sys, wepar, false)
        #     project_to_realspace!(i, j, flux, flxrec, sys, :Y)
        # end

        # AdaWENO scheme
        update_smoothnessfunctions!(smooth, state, sys, wepar.ev)
        for j in gridy.cr_cell, i in gridx.cr_cell
            update_local_smoothnessfunctions!(i, j, smooth, wepar, :X)
            Weno.nonlinear_weights_plus!(wepar)
            Weno.nonlinear_weights_minus!(wepar)
            Weno.update_switches!(wepar)
            if wepar.θp > 0.5 && wepar.θm > 0.5
                update_local!(i, j, state.Q_cons, flux.Fx, q, f, sys, :X)
                Weno.update_numerical_fluxes!(i, j, flux.Fx_hat, q, f, sys, wepar, true)
            else
                update_xeigenvectors!(i, j, state, flxrec, sys)
                project_to_localspace!(i, j, state, flux, flxrec, sys, :X)
                update_local!(i, j, state.Q_proj, flux.Gx, q, f, sys, :X)
                Weno.update_numerical_fluxes!(i, j, flux.Gx_hat, q, f, sys, wepar, false)
                project_to_realspace!(i, j, flux, flxrec, sys, :X)
            end
        end
        for j in gridy.cr_cell, i in gridx.cr_cell
            update_local_smoothnessfunctions!(i, j, smooth, wepar, :Y)
            Weno.nonlinear_weights_plus!(wepar)
            Weno.nonlinear_weights_minus!(wepar)
            Weno.update_switches!(wepar)
            if wepar.θp > 0.5 && wepar.θm > 0.5
                update_local!(i, j, state.Q_cons, flux.Fy, q, f, sys, :Y)
                Weno.update_numerical_fluxes!(i, j, flux.Fy_hat, q, f, sys, wepar, true)
            else
                update_yeigenvectors!(i, j, state, flxrec, sys)
                project_to_localspace!(i, j, state, flux, flxrec, sys, :Y)
                update_local!(i, j, state.Q_proj, flux.Gy, q, f, sys, :Y)
                Weno.update_numerical_fluxes!(i, j, flux.Gy_hat, q, f, sys, wepar, false)
                project_to_realspace!(i, j, flux, flxrec, sys, :Y)
            end
        end

        for j in gridy.cr_mesh, i in gridx.cr_mesh
            update_local_Az_derivatives!(i, j, state, sys, wepar, :X)
            update_numerical_Az_derivatives!(i, j, state, flux, wepar, :X)
        end
        for j in gridy.cr_mesh, i in gridx.cr_mesh
            update_local_Az_derivatives!(i, j, state, sys, wepar, :Y)
            update_numerical_Az_derivatives!(i, j, state, flux, wepar, :Y)
        end

        time_evolution_Az!(state, flux, sys, dt, rApar)
        boundary_conditions_Az!(state, sys, bctype)

        time_evolution!(state, flux, sys, dt, rkpar)
        correct_magneticfield!(state, sys)
        boundary_conditions_conserved!(state, sys, bctype)

        conserved_to_primitive!(state, sys)
        boundary_conditions_primitive!(state, sys, bctype)

        counter += 1
        if counter % 100 == 0
            @printf("Iteration %d: t = %2.3f, dt = %2.3e, v_max = %6.5f, Elapsed time = %3.3f\n", 
                counter, t, dt, wepar.ev, time() - t0)
            # plot_system(state.Q_cons[:, :, 1], sys, "Rho", "orszagtang_rho_256x256_t3_ada")
        end
        # if t > t_array[t_counter]
        #     plot_system(state.Q_cons[:, :, 1], sys, "Rho", "orszagtang_rho_256x256_t$(t_counter)_char")
        #     plot_system(state.Q_prim[:, :, 4], sys, "P", "orszagtang_P_256x256_t$(t_counter)_char")
        #     plot_system(state.Az, sys, "Az", "orszagtang_Az_256x256_t$(t_counter)_char")
        #     plot_system(state.Q_cons[:, :, 5], sys, "Bx", "orszagtang_Bx_256x256_t$(t_counter)_char")
        #     plot_system(state.Q_cons[:, :, 6], sys, "By", "orszagtang_By_256x256_t$(t_counter)_char")
        
        #     T = calculate_temperature(state, sys)
        #     plot_system(T, sys, "T", "orszagtang_T_256x256_t$(t_counter)_char")

        #     divB = calculate_divergence(state, sys)
        #     plot_system(divB, sys, "divB", "orszagtang_divB_256x256_t$(t_counter)_char")

        #     t_counter += 1
        # end
    end

    @printf("%d iterations. t_max = %2.3f. Elapsed time = %3.3f\n", 
        counter, t, time() - t0)
    plot_system(state.Q_cons[:, :, 1], sys, "Rho", "orszagtang_rho_256x256_t4_char")
    plot_system(state.Q_prim[:, :, 4], sys, "P", "orszagtang_P_256x256_t4_char")
    plot_system(state.Az, sys, "Az", "orszagtang_Az_256x256_t4_char")
    plot_system(state.Q_cons[:, :, 5], sys, "Bx", "orszagtang_Bx_256x256_t4_char")
    plot_system(state.Q_cons[:, :, 6], sys, "By", "orszagtang_By_256x256_t4_char")

    T = calculate_temperature(state, sys)
    plot_system(T, sys, "T", "orszagtang_T_256x256_t4_char")
end

@time idealmhd(t_max=0.5)

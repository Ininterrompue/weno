include("./System.jl")
include("./Weno.jl")
using Printf, LinearAlgebra
import Plots, BenchmarkTools
Plots.pyplot()

"""
Primitive: (u, v, w, P, Bx)
Conserved: (ρ, ρu, ρv, ρw, By, Bz, E)
"""
struct StateVectors{T}
    Q_prim::Matrix{T}    # real space primitive variables
    Q_cons::Matrix{T}    # real space conserved, dynamic variables
    Q_proj::Matrix{T}    # characteristic space
    Q_local::Matrix{T}   # local real space
end

struct Fluxes{T}
    Fx::Matrix{T}        # physical flux, real space
    Gx::Matrix{T}        # physical flux, characteristic space
    Fx_hat::Matrix{T}    # numerical flux, real space
    Gx_hat::Matrix{T}    # numerical flux, characteristic space
    F_local::Matrix{T}   # local physical flux, real space
end

mutable struct FluxReconstruction{T}
    Q_avgprim::Matrix{T}   # averaged nonconserved quantities
    Q_avgcons::Matrix{T}   # averaged conserved quantities
    Lx::Matrix{T}          # left eigenvectors
    Rx::Matrix{T}          # right eigenvectors
end

struct SmoothnessFunctions{T}
    G₊::Vector{T}
    G₋::Vector{T}
end


function preallocate_statevectors(sys)
    nx = sys.gridx.nx; nprim = sys.nprim; ncons = sys.ncons
    Q_prim = zeros(nx, nprim)
    Q_cons = zeros(nx, ncons)
    Q_proj = zeros(nx, ncons)
    Q_local = zeros(6, ncons)
    return StateVectors(Q_prim, Q_cons, Q_proj, Q_local)
end

function preallocate_fluxes(sys)
    nx = sys.gridx.nx; ncons = sys.ncons
    Fx = zeros(nx, ncons); Fx_hat = zeros(nx+1, ncons)
    Gx = zeros(nx, ncons); Gx_hat = zeros(nx+1, ncons)
    F_local = zeros(6, ncons)
    return Fluxes(Fx, Gx, Fx_hat, Gx_hat, F_local)
end

function preallocate_fluxreconstruction(sys)
    nx = sys.gridx.nx; nprim = sys.nprim; ncons = sys.ncons
    Q_avgprim = zeros(nx+1, nprim)
    Q_avgcons = zeros(nx+1, ncons)
    evecLx = zeros(ncons, ncons)
    evecRx = zeros(ncons, ncons)
    return FluxReconstruction(Q_avgprim, Q_avgcons, evecLx, evecRx)
end

function preallocate_smoothnessfunctions(sys)
    G₊ = zeros(sys.gridx.nx)
    G₋ = zeros(sys.gridx.nx)
    return SmoothnessFunctions(G₊, G₋)
end

mag2(x, y) = x^2 + y^2
mag2(x, y, z) = x^2 + y^2 + z^2

"""
Brio and Wu Riemann problem 1 (Sod problem with B-field)
x = [-1, 1], t_max = 0.2
"""
function briowu1!(state, sys)
    half = sys.gridx.nx ÷ 2
    Q_prim = state.Q_prim; Q_cons = state.Q_cons
    Q_cons[1:half, 1] .= 1.0;  Q_cons[half+1:end, 1] .= 0.125   # ρ
    Q_prim[1:half, 4] .= 1.0;  Q_prim[half+1:end, 4] .= 0.1     # P
    Q_prim[1:half, 5] .= 0.75; Q_prim[half+1:end, 5] .= 0.75    # Bx
    Q_cons[1:half, 5] .= 1.0;  Q_cons[half+1:end, 5] .= -1.0    # By
end

"""
Brio and Wu Riemann problem 2 (Sod high Mach number flow)
x = [-1, 1], t_max = 0.012
"""
function briowu2!(state, sys)
    half = sys.gridx.nx ÷ 2
    Q_prim = state.Q_prim; Q_cons = state.Q_cons
    Q_cons[1:half, 1] .= 1.0;    Q_cons[half+1:end, 1] .= 0.125   # ρ
    Q_prim[1:half, 4] .= 1000.0; Q_prim[half+1:end, 4] .= 0.1     # P
    Q_prim[1:half, 5] .= 0.75;   Q_prim[half+1:end, 5] .= 0.75    # Bx
    Q_cons[1:half, 5] .= 1.0;    Q_cons[half+1:end, 5] .= -1.0    # By
end

function primitive_to_conserved!(state, sys)
    nx = sys.gridx.nx; γ = sys.γ
    Q_prim = state.Q_prim; Q_cons = state.Q_cons
    for i in 1:nx
        Q_cons[i, 2]  = Q_cons[i, 1] * Q_prim[i, 1]   # ρu = ρ * u
        Q_cons[i, 3]  = Q_cons[i, 1] * Q_prim[i, 2]   # ρv = ρ * v
        Q_cons[i, 4]  = Q_cons[i, 1] * Q_prim[i, 3]   # ρw = ρ * w
        Q_cons[i, 7] = Q_prim[i, 4] / (γ-1) +         # E  = P / (γ-1) + 1/2 * ρU^2 + 1/2 * B^2
            1/2 * Q_cons[i, 1] * (mag2(Q_prim[i, 1], Q_prim[i, 2], Q_prim[i, 3])) +
            1/2 * mag2(Q_prim[i, 5], Q_cons[i, 5], Q_cons[i, 6])
    end
end

function conserved_to_primitive!(state, sys)
    nx = sys.gridx.nx; γ = sys.γ
    Q_prim = state.Q_prim; Q_cons = state.Q_cons
    for i in 1:nx
        Q_prim[i, 1] = Q_cons[i, 2] / Q_cons[i, 1]   # u = ρu / ρ
        Q_prim[i, 2] = Q_cons[i, 3] / Q_cons[i, 1]   # v = ρv / ρ
        Q_prim[i, 3] = Q_cons[i, 4] / Q_cons[i, 1]   # w = ρw / ρ
        Q_prim[i, 4] = (γ-1) * ( Q_cons[i, 7] -      # P = (γ-1) * (E - 1/2 * ρU^2/ρ - 1/2 * B^2)
            1/2 * mag2(Q_cons[i, 2], Q_cons[i, 3], Q_cons[i, 4]) / Q_cons[i, 1] -
            1/2 * mag2(Q_prim[i, 5], Q_cons[i, 5], Q_cons[i, 6]) )
    end
end

function arithmetic_average!(i, state, flxrec, sys)
    Q_prim = state.Q_prim; Q_cons = state.Q_cons
    Q_avgprim = flxrec.Q_avgprim; Q_avgcons = flxrec.Q_avgcons
    for j in 1:sys.nprim
        Q_avgprim[i, j] = 1/2 * (Q_prim[i, j] + Q_prim[i+1, j])
    end
    for j in 1:sys.ncons
        Q_avgcons[i, j] = 1/2 * (Q_cons[i, j] + Q_cons[i+1, j])
    end
end

function slow_magnetosonic(ρ, P, Bx, By, Bz, γ)
    a2 = γ*P/ρ; b2 = (Bx^2 + By^2 + Bz^2)/ρ; bx2 = Bx^2/ρ
    return 1/sqrt(2) * sqrt(a2 + b2 - sqrt((a2 + b2)^2 - 4a2 * bx2))
end

function fast_magnetosonic(ρ, P, Bx, By, Bz, γ)
    a2 = γ*P/ρ; b2 = (Bx^2 + By^2 + Bz^2)/ρ; bx2 = Bx^2/ρ
    return 1/sqrt(2) * sqrt(a2 + b2 + sqrt((a2 + b2)^2 - 4a2 * bx2))
end

function max_eigval(state, sys)
    Q_prim = state.Q_prim; Q_cons = state.Q_cons; γ = sys.γ
    return @views @. $maximum( abs(Q_prim[:, 1]) +
        fast_magnetosonic(Q_cons[:, 1], Q_prim[:, 4], Q_prim[:, 5],
                          Q_cons[:, 5], Q_cons[:, 6], γ) )
end 

CFL_condition(v, cfl, sys) = 0.02 * cfl * sys.gridx.dx / v

function update_physical_fluxes!(flux, state, sys)
    Fx = flux.Fx; nx = sys.gridx.nx
    Q_prim = state.Q_prim; Q_cons = state.Q_cons
    for i in 1:nx
        ρ = Q_cons[i, 1]; ρu = Q_cons[i, 2]; ρv = Q_cons[i, 3]; ρw = Q_cons[i, 4]
        By = Q_cons[i, 5]; Bz = Q_cons[i, 6]; E = Q_cons[i, 7]
        u = Q_prim[i, 1]; v = Q_prim[i, 2]; w = Q_prim[i, 3]
        P = Q_prim[i, 4]; Bx = Q_prim[i, 5]
        P_B = 1/2 * mag2(Bx, By, Bz)

        Fx[i, 1] = ρu
        Fx[i, 2] = ρ * u^2 + P + P_B
        Fx[i, 3] = ρ * u*v - Bx * By
        Fx[i, 4] = ρ * u*w - Bx * Bz
        Fx[i, 5] = u * By - v * Bx
        Fx[i, 6] = u * Bz - w * Bx
        Fx[i, 7] = u * (E + P + P_B) - Bx * (u*Bx + v*By + w*Bz)
    end
end

function update_smoothnessfunctions!(smooth, state, sys, α)
    nx = sys.gridx.nx
    Q_prim = state.Q_prim; Q_cons = state.Q_cons
    for i in 1:nx
        ρ = Q_cons[i, 1]; ρu = Q_cons[i, 2]; ρv = Q_cons[i, 3]; ρw = Q_cons[i, 4]
        P = Q_prim[i, 4]; Bx = Q_prim[i, 5]; By = Q_cons[i, 5]; Bz = Q_cons[i, 6]
        P_B = 1/2 * mag2(Bx, By, Bz)

        smooth.G₊[i] = ρ + mag2(ρu, ρv, ρw) + P + P_B + α * (ρu + ρv + ρw)
        smooth.G₋[i] = ρ + mag2(ρu, ρv, ρw) + P + P_B - α * (ρu + ρv + ρw)
    end
end

function update_xeigenvectors!(i, state, flxrec, sys)
    Lx = flxrec.Lx; Rx = flxrec.Rx; γ = sys.γ
    Q_avgprim = flxrec.Q_avgprim; Q_avgcons = flxrec.Q_avgcons
    
    arithmetic_average!(i, state, flxrec, sys)

    ρ = Q_avgcons[i, 1]; u = Q_avgprim[i, 1]; v = Q_avgprim[i, 2]
    w = Q_avgprim[i, 3]; P = Q_avgprim[i, 4]; Bx = Q_avgprim[i, 5]
    By = Q_avgcons[i, 5]; Bz = Q_avgcons[i, 6]; E = Q_avgcons[i, 7]
    a = sqrt(γ*P/ρ)   # sound speed
    cs = slow_magnetosonic(ρ, P, Bx, By, Bz, γ)
    cf = fast_magnetosonic(ρ, P, Bx, By, Bz, γ)

    if By^2 + Bz^2 > 1e-12 * mag2(Bx, By, Bz)
        βy = By / sqrt(mag2(By, Bz))
        βz = Bz / sqrt(mag2(By, Bz))
    else
        βy = 1/sqrt(2)
        βz = 1/sqrt(2)
    end
    if By^2 + Bz^2 > 1e-12 * mag2(Bx, By, Bz) || abs(γ*P - Bx^2) > 1e-12 * γ * P
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
    Rx[5, 1] = a * αs * βy / sqrt(ρ)
    Rx[6, 1] = a * αs * βz / sqrt(ρ)
    Rx[7, 1] = αf * (1/2 * mag2(u, v, w) + cf^2 - γ2 * a^2) - Γf

    Rx[3, 2] = -βz * sign(Bx)
    Rx[4, 2] = +βy * sign(Bx)
    Rx[5, 2] = -βz / sqrt(ρ)
    Rx[6, 2] = +βy / sqrt(ρ)
    Rx[7, 2] = -Γa

    Rx[1, 3] = αs
    Rx[2, 3] = αs * (u - cs)
    Rx[3, 3] = αs * v - cf * αf * βy * sign(Bx)
    Rx[4, 3] = αs * w - cf * αf * βz * sign(Bx)
    Rx[5, 3] = -a * αf * βy / sqrt(ρ)
    Rx[6, 3] = -a * αf * βz / sqrt(ρ)
    Rx[7, 3] = αs * (1/2 * mag2(u, v, w) + cs^2 - γ2 * a^2) - Γs

    Rx[1, 4] = 1
    Rx[2, 4] = u
    Rx[3, 4] = v
    Rx[4, 4] = w
    Rx[7, 4] = 1/2 * mag2(u, v, w)

    Rx[1, 5] = αs
    Rx[2, 5] = αs * (u + cs)
    Rx[3, 5] = αs * v + cf * αf * βy * sign(Bx)
    Rx[4, 5] = αs * w + cf * αf * βz * sign(Bx)
    Rx[5, 5] = -a * αf * βy / sqrt(ρ)
    Rx[6, 5] = -a * αf * βz / sqrt(ρ)
    Rx[7, 5] = αs * (1/2 * mag2(u, v, w) + cs^2 - γ2 * a^2) + Γs

    Rx[3, 6] = -βz * sign(Bx)
    Rx[4, 6] = +βy * sign(Bx)
    Rx[5, 6] = +βz / sqrt(ρ)
    Rx[6, 6] = -βy / sqrt(ρ)
    Rx[7, 6] = -Γa

    Rx[1, 7] = αf
    Rx[2, 7] = αf * (u + cf)
    Rx[3, 7] = αf * v - cs * αs * βy * sign(Bx)
    Rx[4, 7] = αf * w - cs * αs * βz * sign(Bx)
    Rx[5, 7] = a * αs * βy / sqrt(ρ)
    Rx[6, 7] = a * αs * βz / sqrt(ρ)
    Rx[7, 7] = αf * (1/2 * mag2(u, v, w) + cf^2 - γ2 * a^2) + Γf

    flxrec.Lx = inv(Rx)
    return

    # issues with using L directly
    Lx[1, 1] = oneover2a2 * (γ1 * αf * mag2(u, v, w) + Γf)
    Lx[1, 2] = oneover2a2 * ((1-γ) * αf * u - αf * cf)
    Lx[1, 3] = oneover2a2 * ((1-γ) * αf * v + cs * αs * βy * sign(Bx))
    Lx[1, 4] = oneover2a2 * ((1-γ) * αf * w + cs * αs * βz * sign(Bx))
    Lx[1, 5] = oneover2a2 * ((1-γ) * αf * By - sqrt(ρ) * a * αs * βy)
    Lx[1, 6] = oneover2a2 * ((1-γ) * αf * Bz - sqrt(ρ) * a * αs * βz)
    Lx[1, 7] = oneover2a2 * ((γ-1) * αf)

    Lx[2, 1] = 1/2 * Γa
    Lx[2, 3] = 1/2 * -βz * sign(Bx)
    Lx[2, 4] = 1/2 * +βy * sign(Bx)
    Lx[2, 5] = 1/2 * -sqrt(ρ) * βz
    Lx[2, 6] = 1/2 * +sqrt(ρ) * βy

    Lx[3, 1] = oneover2a2 * (γ1 * αs * mag2(u, v, w) + Γs)
    Lx[3, 2] = oneover2a2 * ((1-γ) * αs * u - αs * cs)
    Lx[3, 3] = oneover2a2 * ((1-γ) * αs * v - cf * αf * βy * sign(Bx))
    Lx[3, 4] = oneover2a2 * ((1-γ) * αs * w - cf * αf * βz * sign(Bx))
    Lx[3, 5] = oneover2a2 * ((1-γ) * αs * By - sqrt(ρ) * a * αf * βy)
    Lx[3, 6] = oneover2a2 * ((1-γ) * αs * Bz - sqrt(ρ) * a * αf * βz)
    Lx[3, 7] = oneover2a2 * ((γ-1) * αs)

    Lx[4, 1] = 1 - 1/2 * τ * mag2(u, v, w)
    Lx[4, 2] = τ * u
    Lx[4, 3] = τ * v
    Lx[4, 4] = τ * w
    Lx[4, 5] = τ * By
    Lx[4, 6] = τ * Bz
    Lx[4, 7] = -τ

    Lx[5, 1] = oneover2a2 * (γ1 * αs * mag2(u, v, w) - Γs)
    Lx[5, 2] = oneover2a2 * ((1-γ) * αs * u + αs * cs)
    Lx[5, 3] = oneover2a2 * ((1-γ) * αs * v + cf * αf * βy * sign(Bx))
    Lx[5, 4] = oneover2a2 * ((1-γ) * αs * w + cf * αf * βz * sign(Bx))
    Lx[5, 5] = oneover2a2 * ((1-γ) * αs * By - sqrt(ρ) * a * αf * βy)
    Lx[5, 6] = oneover2a2 * ((1-γ) * αs * Bz - sqrt(ρ) * a * αf * βz)
    Lx[5, 7] = oneover2a2 * ((γ-1) * αs)

    Lx[6, 1] = 1/2 * Γa
    Lx[6, 3] = 1/2 * -βz * sign(Bx)
    Lx[6, 4] = 1/2 * +βy * sign(Bx)
    Lx[6, 5] = 1/2 * +sqrt(ρ) * βz
    Lx[6, 6] = 1/2 * -sqrt(ρ) * βy

    Lx[7, 1] = oneover2a2 * (γ1 * αf * mag2(u, v, w) - Γf)
    Lx[7, 2] = oneover2a2 * ((1-γ) * αf * u + αf * cf)
    Lx[7, 3] = oneover2a2 * ((1-γ) * αf * v - cs * αs * βy * sign(Bx))
    Lx[7, 4] = oneover2a2 * ((1-γ) * αf * w - cs * αs * βz * sign(Bx))
    Lx[7, 5] = oneover2a2 * ((1-γ) * αf * By - sqrt(ρ) * a * αs * βy)
    Lx[7, 6] = oneover2a2 * ((1-γ) * αf * Bz - sqrt(ρ) * a * αs * βz)
    Lx[7, 7] = oneover2a2 * ((γ-1) * αf)
end

function update_local!(i, Q, F, Q_local, F_local, sys)
    for n in 1:sys.ncons, k in 1:6
        Q_local[k, n] = Q[i-3+k, n]
        F_local[k, n] = F[i-3+k, n]
    end
end

function update_local_smoothnessfunctions!(i, smooth, w)
    for k in 1:6
        w.fp[k] = smooth.G₊[i-3+k]
        w.fm[k] = smooth.G₋[i-3+k]
    end
end

function project_to_localspace!(i, state, flux, flxrec, sys)
    Q_cons = state.Q_cons; Q_proj = state.Q_proj
    Fx = flux.Fx; Gx = flux.Gx; Lx = flxrec.Lx
    for n in 1:sys.ncons, j in i-2:i+3
        Q_proj[j, n] = @views Lx[n, :] ⋅ Q_cons[j, :]
        Gx[j, n] = @views Lx[n, :] ⋅ Fx[j, :]
    end
end

function project_to_realspace!(i, flux, flxrec, sys)
    F_hat = flux.Fx_hat; G_hat = flux.Gx_hat; Rx = flxrec.Rx
    for n in 1:sys.ncons
        F_hat[i, n] = @views Rx[n, :] ⋅ G_hat[i, :]
    end
end

function update_numerical_fluxes!(i, Fx_hat, q, f, sys, wepar, ada)
    for n in 1:sys.ncons
        Fx_hat[i, n] = Weno.update_numerical_flux(@view(q[:, n]), @view(f[:, n]), wepar, ada)
    end
end

function time_evolution!(state, flux, sys, dt, rkpar)
    for n in 1:sys.ncons
        for i in sys.gridx.cr_mesh
            rkpar.op[i] = Weno.weno_scheme(flux.Fx_hat[i, n], flux.Fx_hat[i-1, n], sys.gridx, rkpar)
        end
        Weno.runge_kutta!(@view(state.Q_cons[:, n]), dt, rkpar)
    end
end

function plot_system(state, sys, filename)
    cr = sys.gridx.cr_mesh; x = sys.gridx.x[cr]
    u = state.Q_prim[cr, 1]
    v = state.Q_prim[cr, 2]
    P = state.Q_prim[cr, 4]
    ρ = state.Q_cons[cr, 1]
    By = state.Q_cons[cr, 5]

    plt = Plots.plot(x, ρ, title="1D ideal MHD equations", label="rho")
    Plots.plot!(x, u, label="u")
    Plots.plot!(x, v, label="v")
    Plots.plot!(x, P, label="P")
    Plots.plot!(x, By, label="By")
    display(plt)
    # Plots.pdf(plt, filename)
end


function idealmhd(; γ=2.0, cfl=0.2, t_max=0.0)
    gridx = grid(size=512, min=-1.0, max=1.0)
    sys = SystemParameters1D(gridx, 5, 7, γ)
    rkpar = Weno.preallocate_rungekutta_parameters(gridx)
    wepar = Weno.preallocate_weno_parameters()
    state = preallocate_statevectors(sys)
    flux = preallocate_fluxes(sys)
    flxrec = preallocate_fluxreconstruction(sys)
    smooth = preallocate_smoothnessfunctions(sys)

    briowu1!(state, sys)
    # briowu2!(state, sys)

    primitive_to_conserved!(state, sys)

    t = 0.0; counter = 0; t0 = time()
    q = state.Q_local; f = flux.F_local
    while t < t_max
        update_physical_fluxes!(flux, state, sys)
        wepar.ev = max_eigval(state, sys)
        dt = CFL_condition(wepar.ev, cfl, sys)
        t += dt 
        
        # Component-wise reconstruction
        # for i in gridx.cr_cell
        #     update_local!(i, state.Q_cons, flux.Fx, q, f, sys)
        #     update_numerical_fluxes!(i, flux.Fx_hat, q, f, sys, wepar, false)
        # end

        # Characteristic-wise reconstruction
        # for i in gridx.cr_cell
        #     update_xeigenvectors!(i, state, flxrec, sys)
        #     project_to_localspace!(i, state, flux, flxrec, sys)
        #     update_local!(i, state.Q_proj, flux.Gx, q, f, sys)
        #     update_numerical_fluxes!(i, flux.Gx_hat, q, f, sys, wepar, false)
        #     project_to_realspace!(i, flux, flxrec, sys)
        # end

        # AdaWENO scheme
        update_smoothnessfunctions!(smooth, state, sys, wepar.ev)
        for i in gridx.cr_cell
            update_local_smoothnessfunctions!(i, smooth, wepar)
            Weno.nonlinear_weights_plus!(wepar)
            Weno.nonlinear_weights_minus!(wepar)
            Weno.update_switches!(wepar)
            if wepar.θp > 0.5 && wepar.θm > 0.5
                update_local!(i, state.Q_cons, flux.Fx, q, f, sys)
                update_numerical_fluxes!(i, flux.Fx_hat, q, f, sys, wepar, true)
            else
                update_xeigenvectors!(i, state, flxrec, sys)
                project_to_localspace!(i, state, flux, flxrec, sys)
                update_local!(i, state.Q_proj, flux.Gx, q, f, sys)
                update_numerical_fluxes!(i, flux.Gx_hat, q, f, sys, wepar, false)
                project_to_realspace!(i, flux, flxrec, sys)
            end
        end

        time_evolution!(state, flux, sys, dt, rkpar)
        conserved_to_primitive!(state, sys)

        counter += 1
        if counter % 1000 == 0
            @printf("Iteration %d: t = %2.3f, dt = %2.3e, v_max = %6.5f, Elapsed time = %3.3f\n", 
                counter, t, dt, wepar.ev, time() - t0)
        end
    end

    @printf("%d iterations. t_max = %2.3f. Elapsed time = %3.3f\n", 
        counter, t, time() - t0)
    plot_system(state, sys, "briowu1_512_ada")
end

@time idealmhd(t_max=0.2)

                    
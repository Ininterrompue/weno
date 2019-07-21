# Solves the 2D Euler equations
#   ∂t(Q) + ∂x(Fx) + ∂y(Fy) = 0
# where Q  = [ρ, ρu, ρv, E],
#       Fx = [ρu, ρu² + P, ρuv, u(E + P)]
#       Fy = [ρv, ρvu, ρv² + P, v(E + P)]
# and   E  = P/(γ-1) + 1/2 * ρ(u² + v²).
# For a gas with 5 dofs, γ = 7/5.

include("./Weno.jl")
using Printf, LinearAlgebra
import Plots, BenchmarkTools
Plots.pyplot()

abstract type BoundaryCondition end
struct DoubleMachReflection <: BoundaryCondition end

struct Variables{T}
    ρ::Matrix{T}    # density
    u::Matrix{T}    # velocity
    v::Matrix{T}
    P::Matrix{T}    # pressure
    ρu::Matrix{T}   # momentum
    ρv::Matrix{T}
    E::Matrix{T}    # energy
end

struct ConservedVariables{T}
    ρ::T
    ρu::T
    ρv::T
    E::T
end

struct StateVectors{T}
    Q::Variables{T}                          # real space
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
    Q_avg::Variables{T}   # averaged quantities
    Jx::Matrix{T}         # Jacobians
    Jy::Matrix{T}
    evalx::Vector{T}      # eigenvalues
    evaly::Vector{T}
    evecLx::Matrix{T}     # left eigenvectors
    evecLy::Matrix{T}
    evecRx::Matrix{T}     # right eigenvectors
    evecRy::Matrix{T}
end


function preallocate_statevectors(gridx)
    for x in [:ρ, :u, :v, :P, :ρu, :ρv, :E, :ρ2, :ρu2, :ρv2, :E2]
        @eval $x = zeros($gridx.nx, $gridx.nx)
    end
    for x in [:ρ3, :ρu3, :ρv3, :E3]
        @eval $x = zeros(6)
    end
    Q = Variables(ρ, u, v, P, ρu, ρv, E)
    Q_proj = ConservedVariables(ρ2, ρu2, ρv2, E2)
    Q_local = ConservedVariables(ρ3, ρu3, ρv3, E3)
    return StateVectors(Q, Q_proj, Q_local)
end

function preallocate_fluxes(gridx)
    for x in [:fx1, :fx2, :fx3, :fx4, :fy1, :fy2, :fy3, :fy4,
              :gx1, :gx2, :gx3, :gx4, :gy1, :gy2, :gy3, :gy4]
        @eval $x = zeros($gridx.nx, $gridx.nx)
    end
    for x in [:fx1_hat, :fx2_hat, :fx3_hat, :fx4_hat, :fy1_hat, :fy2_hat, :fy3_hat, :fy4_hat,
              :gx1_hat, :gx2_hat, :gx3_hat, :gx4_hat, :gy1_hat, :gy2_hat, :gy3_hat, :gy4_hat]
        @eval $x = zeros($gridx.nx+1, $gridx.nx+1)
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

function preallocate_fluxreconstruction(gridx)
    for x in [:ρ, :u, :v, :P, :ρu, :ρv, :E]
        @eval $x = zeros($gridx.nx+1, $gridx.nx+1)
    end
    Q_avg = Variables(ρ, u, v, P, ρu, ρv, E)
    for x in [:Jx, :Jy, :evecLx, :evecLy, :evecRx, :evecRy]
        @eval $x = zeros(4, 4)
    end
    evalx = zeros(4); evaly = zeros(4)
    return FluxReconstruction(Q_avg, Jx, Jy, evalx, evaly, 
        evecLx, evecLy, evecRx, evecRy)
end

"""
Case 6 Riemann problem
(x, y) = [0, 1] × [0, 1], t_max = 0.3
"""
function case6!(Q, gridx, gridy)
    halfx = gridx.nx ÷ 2; halfy = gridy.nx ÷ 2

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
function case12!(Q, gridx, gridy)
    halfx = gridx.nx ÷ 2; halfy = gridy.nx ÷ 2

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
function doublemach!(Q, gridx, gridy)
    crx = gridx.cr_mesh; cry = gridy.cr_mesh

    onesixth = argmin(@. abs(gridx - 1/6))
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

function arithmetic_average!(Q, Q_avg, gridx)
    for i in gridx.cr_cell
        Q_avg.ρ[i]  = 1/2 * (Q.ρ[i]  + Q.ρ[i+1])
        Q_avg.u[i]  = 1/2 * (Q.u[i]  + Q.u[i+1])
        Q_avg.v[i]  = 1/2 * (Q.v[i]  + Q.v[i+1])
        Q_avg.P[i]  = 1/2 * (Q.P[i]  + Q.P[i+1])
    end
end

sound_speed(P, ρ, γ) = sqrt(γ*P/ρ)

max_eigval(Q, γ) = @. $maximum(sqrt(Q.u^2 + Q.v^2) + sound_speed(Q.P, Q.ρ, γ))

CFL_condition(eigval, cfl, gridx) = 0.1 * cfl * gridx.dx / eigval

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


# J is defined starting from the leftmost j-1/2.
function update_jacobian!(i, Q, flxrec, γ, gridx)
    Q_avg = flxrec.Q_avg; Jx = flxrec.Jx

    arithmetic_average!(Q, Q_avg, gridx)
    Jx[2, 1] = -(3-γ)/2 * Q_avg.u[i]^2
    Jx[3, 1] = (γ-2)/2 * Q_avg.u[i]^3 - (Q_avg.u[i] * (γ * Q_avg.P[i] / Q_avg.ρ[i]) / (γ-1))
    Jx[1, 2] = 1
    Jx[2, 2] = (3-γ) * Q_avg.u[i]
    Jx[3, 2] = (γ * Q_avg.P[i] / Q_avg.ρ[i]) / (γ-1) + (3-2γ)/2 * Q_avg.u[i]^2
    Jx[2, 3] = γ-1
    Jx[3, 3] = γ * Q_avg.u[i]
end

function update_xlocal!(i, j, Q, F, Q_local, F_local)
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
end

function update_ylocal!(i, j, Q, F, Q_local, F_local)
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

function project_to_localspace!(i, state, flux, flxrec)
    Q = state.Q; Q_proj = state.Q_proj
    F = flux.F; G = flux.G
    L = flxrec.L 
    for k in i-2:i+3
        Q_proj.ρ[k]  = L[1, 1] * Q.ρ[k] + L[1, 2] * Q.ρu[k] + L[1, 3] * Q.E[k]
        Q_proj.ρu[k] = L[2, 1] * Q.ρ[k] + L[2, 2] * Q.ρu[k] + L[2, 3] * Q.E[k]
        Q_proj.E[k]  = L[3, 1] * Q.ρ[k] + L[3, 2] * Q.ρu[k] + L[3, 3] * Q.E[k]
        G.ρ[k]  = L[1, 1] * F.ρ[k] + L[1, 2] * F.ρu[k] + L[1, 3] * F.E[k]
        G.ρu[k] = L[2, 1] * F.ρ[k] + L[2, 2] * F.ρu[k] + L[2, 3] * F.E[k]
        G.E[k]  = L[3, 1] * F.ρ[k] + L[3, 2] * F.ρu[k] + L[3, 3] * F.E[k]
    end
end

function project_to_realspace!(i, flux, flxrec)
    F_hat = flux.F_hat; G_hat = flux.G_hat; R = flxrec.R
    F_hat.ρ[i]  = R[1, 1] * G_hat.ρ[i] + R[1, 2] * G_hat.ρu[i] + R[1, 3] * G_hat.E[i]
    F_hat.ρu[i] = R[2, 1] * G_hat.ρ[i] + R[2, 2] * G_hat.ρu[i] + R[2, 3] * G_hat.E[i]
    F_hat.E[i]  = R[3, 1] * G_hat.ρ[i] + R[3, 2] * G_hat.ρu[i] + R[3, 3] * G_hat.E[i]
end

function boundary_conditions!(Q, bctype::DoubleMachReflection)

end

function plot_system(q, gridx, gridy, titlename, filename)
    crx = gridx.x[gridx.cr_mesh]; cry = gridy.x[gridy.cr_mesh]
    plt = Plots.contour(crx, cry, q[gridx.cr_mesh, gridy.cr_mesh], title=titlename, 
                        fill=true, linecolor=:plasma, levels=15, aspect_ratio=1)
    display(plt)
    # Plots.png(plt, filename)
end

function euler(; γ=7/5, cfl=0.3, t_max=1.0)
    gridx = Weno.grid(size=256, min=0.0, max=1.0)
    gridy = Weno.grid(size=256, min=0.0, max=1.0)
    rkpar = Weno.preallocate_rungekutta_parameters(gridx, gridy)
    wepar = Weno.preallocate_weno_parameters(gridx)
    state = preallocate_statevectors(gridx)
    flux = preallocate_fluxes(gridx)
    flxrec = preallocate_fluxreconstruction(gridx)

    # INITIAL CONDITIONS 
    # doublemach!(state.Q, gridx, gridy)
    case6!(state.Q, gridx, gridy)
    # case12!(state.Q, gridx, gridy)

    primitive_to_conserved!(state.Q, γ)
    t = 0.0; counter = 0; t0 = time()

    q = state.Q_local; f = flux.F_local
    F̂x = flux.Fx_hat; F̂y = flux.Fy_hat
    Ĝx = flux.Gx_hat; Ĝy = flux.Gy_hat
    while t < t_max
        update_physical_fluxes!(flux, state.Q)
        boundary_conditions!(state.Q, DoubleMachReflection())

        wepar.ev = max_eigval(state.Q, γ)
        dt = CFL_condition(wepar.ev, cfl, gridx)
        t += dt 
        
        # Component-wise reconstruction
        for j in gridy.cr_cell, i in gridx.cr_cell
            update_xlocal!(i, j, state.Q, flux.Fx, q, f)
            F̂x.ρ[i, j]  = Weno.update_numerical_flux(q.ρ,  f.ρ,  wepar)
            F̂x.ρu[i, j] = Weno.update_numerical_flux(q.ρu, f.ρu, wepar)
            F̂x.ρv[i, j] = Weno.update_numerical_flux(q.ρv, f.ρv, wepar)
            F̂x.E[i, j]  = Weno.update_numerical_flux(q.E,  f.E,  wepar)
        end
        for j in gridy.cr_cell, i in gridx.cr_cell
            update_ylocal!(i, j, state.Q, flux.Fy, q, f)
            F̂y.ρ[i, j]  = Weno.update_numerical_flux(q.ρ,  f.ρ,  wepar)
            F̂y.ρu[i, j] = Weno.update_numerical_flux(q.ρu, f.ρu, wepar)
            F̂y.ρv[i, j] = Weno.update_numerical_flux(q.ρv, f.ρv, wepar)
            F̂y.E[i, j]  = Weno.update_numerical_flux(q.E,  f.E,  wepar)
        end

        # Characteristic-wise reconstruction
        # for j in gridy.cr_cell, i in gridx.cr_cell
        #     update_xjacobian!(i, j, state.Q, flxrec, γ, gridx)
        #     Weno.diagonalize_jacobian!(flxrec)
        #     project_to_localspace!(i, state, flux, flxrec)
        #     update_local!(i, state.Q_proj, flux.G, q, f)
        #     Ĝx.ρ[i]  = Weno.update_numerical_flux(q.ρ,  f.ρ,  wepar)
        #     Ĝx.ρu[i] = Weno.update_numerical_flux(q.ρu, f.ρu, wepar)
        #     Ĝx.E[i]  = Weno.update_numerical_flux(q.E,  f.E,  wepar)
        #     project_to_realspace!(i, flux, flxrec)
        # end

        Weno.weno_scheme!(F̂x.ρ, F̂y.ρ, gridx, gridy, rkpar)
        Weno.runge_kutta!(state.Q.ρ, dt, rkpar)

        Weno.weno_scheme!(F̂x.ρu, F̂y.ρu, gridx, gridy, rkpar)
        Weno.runge_kutta!(state.Q.ρu, dt, rkpar)

        Weno.weno_scheme!(F̂x.ρv, F̂y.ρv, gridx, gridy, rkpar)
        Weno.runge_kutta!(state.Q.ρv, dt, rkpar)

        Weno.weno_scheme!(F̂x.E, F̂y.E, gridx, gridy, rkpar)
        Weno.runge_kutta!(state.Q.E, dt, rkpar)

        conserved_to_primitive!(state.Q, γ)

        counter += 1
        if counter % 100 == 0
            @printf("Iteration %d: t = %2.3f, dt = %2.3e, elapsed = %3.3f\n", 
                counter, t, dt, time() - t0)
            # plot_system(state.Q.ρ, gridx, gridy, "Mass density", "euler2d_case12_rho_128x128")
        end
    end

    @printf("%d iterations. t_max = %2.3f.\n", counter, t)
    plot_system(state.Q.ρ, gridx, gridy, "Mass density", "euler2d_case12_rho_128x128")
    plot_system(state.Q.P, gridx, gridy, "Pressure", "euler2d_case12_P_128x128")
end

# BenchmarkTools.@btime euler(t_max=0.01);
@time euler(t_max=0.3)
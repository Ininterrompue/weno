module Weno

export grid, diagonalize_jacobian!, update_numerical_flux
export preallocate_rungekutta_parameters, preallocate_weno_parameters
export weno_scheme!, runge_kutta!

using LinearAlgebra

struct GridParameters
    nx::Int
    dx::Float64
    x::StepRangeLen{Float64, Float64, Float64}
    cr_mesh::UnitRange{Int}
    cr_cell::UnitRange{Int}
    ghost::Int
end

struct RungeKuttaParameters{T}
    op::T   # du/dt = op(u, x) (nonlinear operator)
    u1::T   # 1st RK-3 iteration
    u2::T   # 2nd RK-3 iteration
    u3::T   # 3rd RK-3 iteration
end

mutable struct WenoParameters{T}
    fp::Vector{T}
    fm::Vector{T}
    ev::T
    fhat0::T
    fhat1::T
    fhat2::T
    w0::T
    w1::T
    w2::T
    IS0::T
    IS1::T
    IS2::T
    α0::T
    α1::T
    α2::T
    τ::T
    ϵ::T
end


function grid(; size=32, min=-1.0, max=1.0, ghost=3)
    nx = 2*ghost + size
    dx = (max-min)/size
    x  = min - (ghost - 1/2)*dx : dx : max + (ghost - 1/2)*dx
    cr_mesh = ghost+1:nx-ghost
    cr_cell = ghost:nx-ghost
    return GridParameters(nx, dx, x, cr_mesh, cr_cell, ghost)
end

function preallocate_rungekutta_parameters(gridx)
    for x in [:op, :u1, :u2, :u3]
        @eval $x = zeros($gridx.nx)
    end
    return RungeKuttaParameters(op, u1, u2, u3)
end

function preallocate_rungekutta_parameters(gridx, gridy)
    for x in [:op, :u1, :u2, :u3]
        @eval $x = zeros($gridx.nx, $gridy.nx)
    end
    return RungeKuttaParameters(op, u1, u2, u3)
end

function preallocate_weno_parameters(gridx)
    fp = zeros(gridx.nx)
    fm = zeros(gridx.nx)
    return WenoParameters(fp, fm, 0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-6)
end

# Need to figure out a way to bypass the inv()
function diagonalize_jacobian!(flxrec)
    flxrec.eval, flxrec.R = eigen(flxrec.J)
    flxrec.L = inv(flxrec.R)
end

function runge_kutta!(u, dt, rkpar)
    @. rkpar.u1 = u + dt * rkpar.op
    @. rkpar.u2 = 3/4 * u + 1/4 * rkpar.u1 + 1/4 * dt * rkpar.op
    @. rkpar.u3 = 1/3 * u + 2/3 * rkpar.u2 + 2/3 * dt * rkpar.op
    @. u = rkpar.u3
end

function weno_scheme!(f_hat, gridx, rkpar)
    for i in gridx.cr_mesh
        rkpar.op[i] = -1/gridx.dx * (f_hat[i] - f_hat[i-1])
    end
end

function weno_scheme!(fx_hat, fy_hat, gridx, gridy, rkpar)
    for j in gridy.cr_mesh, i in gridx.cr_mesh
        rkpar.op[i, j] = 
            -1/gridx.dx * (fx_hat[i, j] - fx_hat[i-1, j]) +
            -1/gridy.dx * (fy_hat[i, j] - fy_hat[i, j-1])
    end
end

# Lax-Friderichs flux splitting
fplus(u, f, ev)  = 1/2 * (f + ev * u)
fminus(u, f, ev) = 1/2 * (f - ev * u)

function update_numerical_flux(u, f, w)
    for i in eachindex(u)
        w.fp[i] = fplus(u[i], f[i], w.ev)
        w.fm[i] = fminus(u[i], f[i], w.ev)
    end
    return fhat(:+, w) + fhat(:-, w)
end

function fhat(flux_sign, w)
    fp = w.fp; fm = w.fm

    if flux_sign == :+
        w.fhat0 =  1/3 * fp[1] - 7/6 * fp[2] + 11/6 * fp[3]
        w.fhat1 = -1/6 * fp[2] + 5/6 * fp[3] +  1/3 * fp[4]
        w.fhat2 =  1/3 * fp[3] + 5/6 * fp[4] -  1/6 * fp[5]
    elseif flux_sign == :-
        w.fhat0 =  1/3 * fm[4] + 5/6 * fm[3] -  1/6 * fm[2]
        w.fhat1 = -1/6 * fm[5] + 5/6 * fm[4] +  1/3 * fm[3]
        w.fhat2 =  1/3 * fm[6] - 7/6 * fm[5] + 11/6 * fm[4]
    end

    nonlinear_weights!(flux_sign, w)
    return w.w0 * w.fhat0 + w.w1 * w.fhat1 + w.w2 * w.fhat2
end

function nonlinear_weights!(flux_sign, w)
    fp = w.fp; fm = w.fm

    if flux_sign == :+
        w.IS0 = 13/12 * (fp[1] - 2fp[2] + fp[3])^2 + 1/4 * (fp[1] - 4fp[2] + 3fp[3])^2
        w.IS1 = 13/12 * (fp[2] - 2fp[3] + fp[4])^2 + 1/4 * (fp[2] - fp[4])^2
        w.IS2 = 13/12 * (fp[3] - 2fp[4] + fp[5])^2 + 1/4 * (3fp[3] - 4fp[4] + fp[5])^2

        # Yamaleev and Carpenter, 2009
        w.τ = (fp[1] - 4fp[2] + 6fp[3] - 4fp[4] + fp[5])^2
        w.α0 = 1/10 * (1 + (w.τ / (w.ϵ + w.IS0))^2)
        w.α1 = 6/10 * (1 + (w.τ / (w.ϵ + w.IS1))^2)
        w.α2 = 3/10 * (1 + (w.τ / (w.ϵ + w.IS2))^2)

        # Jiang and Shu, 1997
        # w.α0 = 1/10 / (w.ϵ + w.IS0)^2
        # w.α1 = 6/10 / (w.ϵ + w.IS1)^2
        # w.α2 = 3/10 / (w.ϵ + w.IS2)^2

    elseif flux_sign == :-
        w.IS0 = 13/12 * (fm[2] - 2fm[3] + fm[4])^2 + 1/4 * (fm[2] - 4fm[3] + 3fm[4])^2
        w.IS1 = 13/12 * (fm[3] - 2fm[4] + fm[5])^2 + 1/4 * (fm[3] - fm[5])^2
        w.IS2 = 13/12 * (fm[4] - 2fm[5] + fm[6])^2 + 1/4 * (3fm[4] - 4fm[5] + fm[6])^2

        w.τ = (fm[2] - 4fm[3] + 6fm[4] - 4fm[5] + fm[6])^2
        w.α0 = 3/10 * (1 + (w.τ / (w.ϵ + w.IS0))^2)
        w.α1 = 6/10 * (1 + (w.τ / (w.ϵ + w.IS1))^2)
        w.α2 = 1/10 * (1 + (w.τ / (w.ϵ + w.IS2))^2)

        # w.α0 = 3/10 / (w.ϵ + w.IS0)^2
        # w.α1 = 6/10 / (w.ϵ + w.IS1)^2
        # w.α2 = 1/10 / (w.ϵ + w.IS2)^2
    end

    w.w0 = w.α0 / (w.α0 + w.α1 + w.α2)
    w.w1 = w.α1 / (w.α0 + w.α1 + w.α2)
    w.w2 = w.α2 / (w.α0 + w.α1 + w.α2)
end

end

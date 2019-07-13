module Weno

export grid, time_evolution!, diagonalize_jacobian!
export preallocate_rungekutta_parameters, preallocate_weno_parameters

using LinearAlgebra

struct GridParameters
    nx::Int
    dx::Float64
    x::StepRangeLen{Float64, Float64, Float64}
    cr::UnitRange{Int}
    ghost::Int
end

struct RungeKuttaParameters{T}
    op::Vector{T}  # du/dt = op(u, x) (nonlinear operator)
    u1::Vector{T}  # 1st RK-3 iteration
    u2::Vector{T}  # 2nd RK-3 iteraiton
    u3::Vector{T}  # 3rd RK-3 iteration
end

mutable struct WenoParameters{T}
    fp::Vector{T}
    fm::Vector{T}
    fp_local::Vector{T}
    fm_local::Vector{T}
    fp_local2::Vector{T}
    fm_local2::Vector{T}
    ev::T
    df::T
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
    cr = 1+ghost:nx-ghost
    return GridParameters(nx, dx, x, cr, ghost)
end

function preallocate_rungekutta_parameters(grpar)
    op = zeros(grpar.nx); u1 = zeros(grpar.nx)
    u2 = zeros(grpar.nx); u3 = zeros(grpar.nx)
    return RungeKuttaParameters(op, u1, u2, u3)
end

function preallocate_weno_parameters(grpar)
    fp = zeros(grpar.nx)
    fm = zeros(grpar.nx)
    fp_local = zeros(2grpar.ghost+1)
    fm_local = zeros(2grpar.ghost+1)
    fp_local2 = zeros(2grpar.ghost)
    fm_local2 = zeros(2grpar.ghost)
    return WenoParameters(fp, fm, fp_local, fm_local, fp_local2, fm_local2,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-6)
end

function diagonalize_jacobian!(U_avg)
    for i in U_avg.cr
        U_avg.evalRe[:, i], U_avg.evalIm[:, i], 
        U_avg.evecL[:, :, i], U_avg.evecR[:, :, i] =
            LAPACK.geev!('V', 'V', U_avg.J[:, :, i])
    end
end

# Lax-Friderichs flux splitting
fplus(u, f, ev)  = 1/2 * (f + ev * u)
fminus(u, f, ev) = 1/2 * (f - ev * u)

function time_evolution!(u, f, dt, grpar, rkpar, wepar)
    @. wepar.fp = fplus(u, f, wepar.ev)
    @. wepar.fm = fminus(u, f, wepar.ev)

    for i in grpar.cr
        for j in 1:7
            wepar.fp_local[j] = wepar.fp[i-3+j-1]
            wepar.fm_local[j] = wepar.fm[i-3+j-1]
        end
        rkpar.op[i] = weno_scheme(grpar, wepar)
    end
    runge_kutta!(u, dt, rkpar)
end

function runge_kutta!(u, dt, rkpar)
    @. rkpar.u1 = u + dt * rkpar.op
    @. rkpar.u2 = 3/4 * u + 1/4 * rkpar.u1 + 1/4 * dt * rkpar.op
    @. rkpar.u3 = 1/3 * u + 2/3 * rkpar.u2 + 2/3 * dt * rkpar.op
    @. u = rkpar.u3
end

# 5th order WENO finite-difference scheme
function weno_scheme(grpar, wepar)
    wepar.df = -1/grpar.dx *
        (fhat(:+, "j+1/2", wepar) + fhat(:-, "j+1/2", wepar) -
         fhat(:+, "j-1/2", wepar) - fhat(:-, "j-1/2", wepar))
    return wepar.df
end

function fhat(flux_sign, half_sign, w)
    if half_sign == "j+1/2"
        for i in 1:6
            w.fp_local2[i] = w.fp_local[i+1]
            w.fm_local2[i] = w.fm_local[i+1]
        end
    elseif half_sign == "j-1/2"
        for i in 1:6
            w.fp_local2[i] = w.fp_local[i]
            w.fm_local2[i] = w.fm_local[i]
        end
    end

    fp = w.fp_local2
    fm = w.fm_local2

    if flux_sign == :+
        w.fhat0 =  1/3 * fp[1] - 7/6 * fp[2] + 11/6 * fp[3]
        w.fhat1 = -1/6 * fp[2] + 5/6 * fp[3] +  1/3 * fp[4]
        w.fhat2 =  1/3 * fp[3] + 5/6 * fp[4] -  1/6 * fp[5]
    elseif flux_sign == :-
        w.fhat0 =  1/3 * fm[4] + 5/6 * fm[3] -  1/6 * fm[2]
        w.fhat1 = -1/6 * fm[5] + 5/6 * fm[4] +  1/3 * fm[3]
        w.fhat2 =  1/3 * fm[6] - 7/6 * fm[5] + 11/6 * fm[4]
    end

    nonlinear_weights!(flux_sign, fp, fm, w)
    return w.w0 * w.fhat0 + w.w1 * w.fhat1 + w.w2 * w.fhat2
end

function nonlinear_weights!(flux_sign, fp, fm, w)
    if flux_sign == :+
        w.IS0 = 13/12 * (fp[1] - 2 * fp[2] + fp[3])^2 +
                  1/4 * (fp[1] - 4 * fp[2] + 3 * fp[3])^2
        w.IS1 = 13/12 * (fp[2] - 2 * fp[3] + fp[4])^2 +
                  1/4 * (fp[2] - fp[4])^2
        w.IS2 = 13/12 * (fp[3] - 2 * fp[4] + fp[5])^2 +
                  1/4 * (3 * fp[3] - 4 * fp[4] + fp[5])^2

        # Yamaleev and Carpenter, 2009
        w.τ = (fp[1] - 4 * fp[2] + 6 * fp[3] - 4 * fp[4] + fp[5])^2
        w.α0 = 1/10 * (1 + (w.τ / (w.ϵ + w.IS0))^2)
        w.α1 = 6/10 * (1 + (w.τ / (w.ϵ + w.IS1))^2)
        w.α2 = 3/10 * (1 + (w.τ / (w.ϵ + w.IS2))^2)

        # Jiang and Shu, 1997
        # w.α0 = 1/10 / (w.ϵ + w.IS0)^2
        # w.α1 = 6/10 / (w.ϵ + w.IS1)^2
        # w.α2 = 3/10 / (w.ϵ + w.IS2)^2

    elseif flux_sign == :-
        w.IS0 = 13/12 * (fm[2] - 2 * fm[3] + fm[4])^2 +
                  1/4 * (fm[2] - 4 * fm[3] + 3 * fm[4])^2
        w.IS1 = 13/12 * (fm[3] - 2 * fm[4] + fm[5])^2 +
                  1/4 * (fm[3] - fm[5])^2
        w.IS2 = 13/12 * (fm[4] - 2 * fm[5] + fm[6])^2 +
                  1/4 * (3 * fm[4] - 4 * fm[5] + fm[6])^2

        w.τ = (fm[2] - 4 * fm[3] + 6 * fm[4] - 4 * fm[5] + fm[6])^2
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

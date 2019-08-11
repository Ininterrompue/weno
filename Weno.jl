module Weno

export diagonalize_jacobian!, update_numerical_flux
export nonlinear_weights_plus!, nonlinear_weights_minus!, update_switches!
export preallocate_rungekutta_parameters, preallocate_weno_parameters
export weno_scheme!, runge_kutta!

using LinearAlgebra


mutable struct RungeKuttaParameters{T}
    op::T   # du/dt = op(u, x) (nonlinear operator)
    u1::T   # 1st RK-3 iteration
    u2::T   # 2nd RK-3 iteration
    u3::T   # 3rd RK-3 iteration
end

mutable struct WenoParameters{T}
    fp::Vector{T}
    fm::Vector{T}
    ev::T
    fhat0p::T
    fhat1p::T
    fhat2p::T
    fhat0m::T
    fhat1m::T
    fhat2m::T
    w0p::T
    w1p::T
    w2p::T
    w0m::T
    w1m::T
    w2m::T
    β0p::T
    β1p::T
    β2p::T
    β0m::T
    β1m::T
    β2m::T
    α0p::T
    α1p::T
    α2p::T
    α0m::T
    α1m::T
    α2m::T
    τp::T
    τm::T
    θp::T
    θm::T
    ϵ::T
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

function preallocate_weno_parameters()
    fp = zeros(6)
    fm = zeros(6)
    return WenoParameters(fp, fm, 0.0,   # ev
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,    # f_hat
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,    # ω
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,    # β
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,    # α
        0.0, 0.0, 0.0, 0.0, 1e-6)        # τ, θ, ϵ
end

# Need to figure out a way to bypass the inv().
# LAPACK.geev!('V', 'V', J) gives wrong results...
function diagonalize_jacobian!(flxrec)
    flxrec.eval, flxrec.R = eigen(flxrec.J)
    flxrec.L = inv(flxrec.R)
end

function update_switches!(w)
    w.θp = 1 / (1 + (w.α0p + w.α1p + w.α2p - 1)^2)
    w.θm = 1 / (1 + (w.α0m + w.α1m + w.α2m - 1)^2)
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

function weno_scheme(f1, f0, gridx, rkpar)
    return -1/gridx.dx * (f1 - f0)
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

function update_numerical_flux(u, f, w, ada)
    @. w.fp = fplus(u, f, w.ev)
    @. w.fm = fminus(u, f, w.ev)
    return fhatp(w, ada) + fhatm(w, ada)
end

function fhatp(w, ada)
    w.fhat0p =  1/3 * w.fp[1] - 7/6 * w.fp[2] + 11/6 * w.fp[3]
    w.fhat1p = -1/6 * w.fp[2] + 5/6 * w.fp[3] +  1/3 * w.fp[4]
    w.fhat2p =  1/3 * w.fp[3] + 5/6 * w.fp[4] -  1/6 * w.fp[5]

    if ada == false nonlinear_weights_plus!(w) end
    return w.w0p * w.fhat0p + w.w1p * w.fhat1p + w.w2p * w.fhat2p
end

function fhatm(w, ada)
    w.fhat0m =  1/3 * w.fm[4] + 5/6 * w.fm[3] -  1/6 * w.fm[2]
    w.fhat1m = -1/6 * w.fm[5] + 5/6 * w.fm[4] +  1/3 * w.fm[3]
    w.fhat2m =  1/3 * w.fm[6] - 7/6 * w.fm[5] + 11/6 * w.fm[4]

    if ada == false nonlinear_weights_minus!(w) end
    return w.w0m * w.fhat0m + w.w1m * w.fhat1m + w.w2m * w.fhat2m
end

function nonlinear_weights_plus!(w)
    w.β0p = 13/12 * (w.fp[1] - 2w.fp[2] + w.fp[3])^2 + 1/4 * (w.fp[1] - 4w.fp[2] + 3w.fp[3])^2
    w.β1p = 13/12 * (w.fp[2] - 2w.fp[3] + w.fp[4])^2 + 1/4 * (w.fp[2] - w.fp[4])^2
    w.β2p = 13/12 * (w.fp[3] - 2w.fp[4] + w.fp[5])^2 + 1/4 * (3w.fp[3] - 4w.fp[4] + w.fp[5])^2

    # Yamaleev and Carpenter, 2009
    w.τp = (w.fp[1] - 4w.fp[2] + 6w.fp[3] - 4w.fp[4] + w.fp[5])^2
    w.α0p = 1/10 * (1 + (w.τp / (w.ϵ + w.β0p))^2)
    w.α1p = 6/10 * (1 + (w.τp / (w.ϵ + w.β1p))^2)
    w.α2p = 3/10 * (1 + (w.τp / (w.ϵ + w.β2p))^2)

    # Jiang and Shu, 1997
    # w.α0p = 1/10 / (w.ϵ + w.β0p)^2
    # w.α1p = 6/10 / (w.ϵ + w.β1p)^2
    # w.α2p = 3/10 / (w.ϵ + w.β2p)^2

    w.w0p = w.α0p / (w.α0p + w.α1p + w.α2p)
    w.w1p = w.α1p / (w.α0p + w.α1p + w.α2p)
    w.w2p = w.α2p / (w.α0p + w.α1p + w.α2p)
end

function nonlinear_weights_minus!(w)
    w.β0m = 13/12 * (w.fm[2] - 2w.fm[3] + w.fm[4])^2 + 1/4 * (w.fm[2] - 4w.fm[3] + 3w.fm[4])^2
    w.β1m = 13/12 * (w.fm[3] - 2w.fm[4] + w.fm[5])^2 + 1/4 * (w.fm[3] - w.fm[5])^2
    w.β2m = 13/12 * (w.fm[4] - 2w.fm[5] + w.fm[6])^2 + 1/4 * (3w.fm[4] - 4w.fm[5] + w.fm[6])^2

    w.τm = (w.fm[2] - 4w.fm[3] + 6w.fm[4] - 4w.fm[5] + w.fm[6])^2
    w.α0m = 3/10 * (1 + (w.τm / (w.ϵ + w.β0m))^2)
    w.α1m = 6/10 * (1 + (w.τm / (w.ϵ + w.β1m))^2)
    w.α2m = 1/10 * (1 + (w.τm / (w.ϵ + w.β2m))^2)

    # w.α0m = 3/10 / (w.ϵ + w.IS0)^2
    # w.α1m = 6/10 / (w.ϵ + w.IS1)^2
    # w.α2m = 1/10 / (w.ϵ + w.IS2)^2

    w.w0m = w.α0m / (w.α0m + w.α1m + w.α2m)
    w.w1m = w.α1m / (w.α0m + w.α1m + w.α2m)
    w.w2m = w.α2m / (w.α0m + w.α1m + w.α2m)
end

end

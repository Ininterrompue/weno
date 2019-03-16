module WenoChar

export grid, runge_kutta
export preallocate_rungekutta_parameters, preallocate_weno_parameters

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


function grid(size=32, min=-1.0, max=1.0, ghost=3)
    nx = 2*ghost + size
    dx = (max-min)/size
    x  = min - (ghost - 1/2)*dx : dx : max + (ghost - 1/2)*dx
    cr = 1+ghost:nx-ghost
    return GridParameters(nx, dx, x, cr, ghost)
end

function preallocate_rungekutta_parameters(grpar)
    nx = grpar.nx
    op = zeros(nx); u1 = zeros(nx)
    u2 = zeros(nx); u3 = zeros(nx)
    return RungeKuttaParameters(op, u1, u2, u3)
end

function preallocate_weno_parameters(grpar)
    fp = zeros(grpar.nx)
    fm = zeros(grpar.nx)
    fp_local = zeros(2grpar.ghost+1)
    fm_local = zeros(2grpar.ghost+1)
    return WenoParameters(fp, fm, fp_local, fm_local, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-6)
end

# 3rd order TVD Runge-Kutta discretizes
#   du/dt = op(u, x)
# where op is a (nonlinear) operator.
function runge_kutta!(u, f, dt, grpar, rkpar, wspar)
    for i in grpar.cr
        wspar.fp[i] = fplus(u[i], f[i], wspar.ev)
        wspar.fm[i] = fminus(u[i], f[i], wspar.ev)
    end
    for i in grpar.cr
        @views wspar.fp_local = wspar.fp[i-3: i+3]
        @views wspar.fm_local = wspar.fm[i-3: i+3]
        rkpar.op[i] = weno_dv(grpar, wspar)
    end

    @. rkpar.u1 = u + dt * rkpar.op
    @. rkpar.u2 = 3/4 * u + 1/4 * rkpar.u1 + 1/4 * dt * rkpar.op
    @. rkpar.u3 = 1/3 * u + 2/3 * rkpar.u2 + 2/3 * dt * rkpar.op
    @. u = rkpar.u3
end

# 5th order WENO finite-difference scheme
function weno_dv(grpar, wspar)
    wspar.df = -1/grpar.dx *
        (fhat("positive", "j+1/2", wspar) + fhat("negative", "j+1/2", wspar) -
         fhat("positive", "j-1/2", wspar) - fhat("negative", "j-1/2", wspar))
    return wspar.df
end

function fhat(flux_sign, half_sign, w)
    if half_sign == "j+1/2"
        @views fp = w.fp_local[2:end]
        @views fm = w.fm_local[2:end]
    elseif half_sign == "j-1/2"
        @views fp = w.fp_local[1:end-1]
        @views fm = w.fm_local[1:end-1]
    end

    if flux_sign == "positive"
        w.fhat0 =  1/3 * fp[1] - 7/6 * fp[2] + 11/6 * fp[3]
        w.fhat1 = -1/6 * fp[2] + 5/6 * fp[3] +  1/3 * fp[4]
        w.fhat2 =  1/3 * fp[3] + 5/6 * fp[4] -  1/6 * fp[5]
    elseif flux_sign == "negative"
        w.fhat0 =  1/3 * fm[4] + 5/6 * fm[3] -  1/6 * fm[2]
        w.fhat1 = -1/6 * fm[5] + 5/6 * fm[4] +  1/3 * fm[3]
        w.fhat2 =  1/3 * fm[6] - 7/6 * fm[5] + 11/6 * fm[4]
    end

    w.w0, w.w1, w.w2 = weights(flux_sign, fp, fm, w)
    return w.w0 * w.fhat0 + w.w1 * w.fhat1 + w.w2 * w.fhat2
end

function weights(flux_sign, fp, fm, w)
    if flux_sign == "positive"
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
        # α0 = 1/10 / (w.ϵ + w.IS0)^2
        # α1 = 6/10 / (w.ϵ + w.IS1)^2
        # α2 = 3/10 / (w.ϵ + w.IS2)^2

    elseif flux_sign == "negative"
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

        # α0 = 3/10 / (w.ϵ + w.IS0)^2
        # α1 = 6/10 / (w.ϵ + w.IS1)^2
        # α2 = 1/10 / (w.ϵ + w.IS2)^2
    end

    w.w0 = w.α0 / (w.α0 + w.α1 + w.α2)
    w.w1 = w.α1 / (w.α0 + w.α1 + w.α2)
    w.w2 = w.α2 / (w.α0 + w.α1 + w.α2)

    return w.w0, w.w1, w.w2
end

# Lax-Friderichs flux splitting
fplus(u, f, ev)  = 1/2 * (f + ev * u)
fminus(u, f, ev) = 1/2 * (f - ev * u)

end

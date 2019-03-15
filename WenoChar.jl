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

mutable struct WenoStaticParameters{T}
    fp::Vector{T}
    fm::Vector{T}
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
    n = length(collect(grpar.x))
    for i in [:op, :u1, :u2, :u3]
        @eval $i = zeros($n)
    end
    return RungeKuttaParameters(op, u1, u2, u3)
end

function preallocate_weno_parameters(grpar)
    fp = fm = zeros(2grpar.ghost+1)
    return WenoStaticParameters(fp, fm, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

# 3rd order TVD Runge-Kutta discretizes
#   du/dt = op(u, x)
# where op is a (nonlinear) operator.
function runge_kutta!(u, f, dt, grpar, rkpar, wspar)
    for i in grpar.cr
        @views u_local = u[i-3: i+3]
        @views f_local = f[i-3: i+3]
        rkpar.op[i] = weno_dv(u_local, f_local, grpar, wspar)
    end

    for i in grpar.cr
        rkpar.u1[i] = u[i] + dt * rkpar.op[i]
        rkpar.u2[i] = 3/4 * u[i] + 1/4 * rkpar.u1[i] + 1/4 * dt * rkpar.op[i]
        rkpar.u3[i] = 1/3 * u[i] + 2/3 * rkpar.u2[i] + 2/3 * dt * rkpar.op[i]
        u[i] = rkpar.u3[i]
    end
end

# 5th order WENO finite-difference scheme
function weno_dv(u_local, f_local, grpar, w)
    @views u_plushalf  = u_local[2:end]
    @views u_minushalf = u_local[1:end-1]
    @views f_plushalf  = f_local[2:end]
    @views f_minushalf = f_local[1:end-1]

    w.df = -1/grpar.dx *
           (fhat(+1, u_plushalf, f_plushalf, w) +
            fhat(-1, u_plushalf, f_plushalf, w) -
            fhat(+1, u_minushalf, f_minushalf, w) -
            fhat(-1, u_minushalf, f_minushalf, w))
    return w.df
end

function fhat(pm, u, f, w)
    for i in eachindex(u)
        w.fp[i] = fplus(u[i], f[i], w)
        w.fm[i] = fminus(u[i], f[i], w)
    end

    # pm: positive/negative flux.
    if pm == 1
        w.fhat0 =  1/3 * w.fp[1] - 7/6 * w.fp[2] + 11/6 * w.fp[3]
        w.fhat1 = -1/6 * w.fp[2] + 5/6 * w.fp[3] +  1/3 * w.fp[4]
        w.fhat2 =  1/3 * w.fp[3] + 5/6 * w.fp[4] -  1/6 * w.fp[5]
        w.w0, w.w1, w.w2 = weights(+1, w)

   elseif pm == -1
        w.fhat0 =  1/3 * w.fm[4] + 5/6 * w.fm[3] -  1/6 * w.fm[2]
        w.fhat1 = -1/6 * w.fm[5] + 5/6 * w.fm[4] +  1/3 * w.fm[3]
        w.fhat2 =  1/3 * w.fm[6] - 7/6 * w.fm[5] + 11/6 * w.fm[4]
        w.w0, w.w1, w.w2 = weights(-1, w)
    end

    return w.w0 * w.fhat0 + w.w1 * w.fhat1 + w.w2 * w.fhat2
end

function weights(pm, w)
    if pm == 1
        w.IS0 = 13/12 * (w.fp[1] - 2 * w.fp[2] + w.fp[3])^2 +
                  1/4 * (w.fp[1] - 4 * w.fp[2] + 3 * w.fp[3])^2
        w.IS1 = 13/12 * (w.fp[2] - 2 * w.fp[3] + w.fp[4])^2 +
                  1/4 * (w.fp[2] - w.fp[4])^2
        w.IS2 = 13/12 * (w.fp[3] - 2 * w.fp[4] + w.fp[5])^2 +
                  1/4 * (3 * w.fp[3] - 4 * w.fp[4] + w.fp[5])^2

        # Yamaleev and Carpenter, 2009
        w.τ = (w.fp[1] - 4 * w.fp[2] + 6 * w.fp[3] - 4 * w.fp[4] + w.fp[5])^2
        w.α0 = 1/10 * (1 + (w.τ / (w.ϵ + w.IS0))^2)
        w.α1 = 6/10 * (1 + (w.τ / (w.ϵ + w.IS1))^2)
        w.α2 = 3/10 * (1 + (w.τ / (w.ϵ + w.IS2))^2)

        # Jiang and Shu, 1997
        # α0 = 1/10 / (ϵ + IS0)^2
        # α1 = 6/10 / (ϵ + IS1)^2
        # α2 = 3/10 / (ϵ + IS2)^2

    elseif pm == -1
        w.IS0 = 13/12 * (w.fm[2] - 2 * w.fm[3] + w.fm[4])^2 +
                  1/4 * (w.fm[2] - 4 * w.fm[3] + 3 * w.fm[4])^2
        w.IS1 = 13/12 * (w.fm[3] - 2 * w.fm[4] + w.fm[5])^2 +
                  1/4 * (w.fm[3] - w.fm[5])^2
        w.IS2 = 13/12 * (w.fm[4] - 2 * w.fm[5] + w.fm[6])^2 +
                  1/4 * (3 * w.fm[4] - 4 * w.fm[5] + w.fm[6])^2

        w.τ = (w.fm[2] - 4 * w.fm[3] + 6 * w.fm[4] - 4 * w.fm[5] + w.fm[6])^2
        w.α0 = 3/10 * (1 + (w.τ / (w.ϵ + w.IS0))^2)
        w.α1 = 6/10 * (1 + (w.τ / (w.ϵ + w.IS1))^2)
        w.α2 = 1/10 * (1 + (w.τ / (w.ϵ + w.IS2))^2)

        # α0 = 3/10 / (ϵ + IS0)^2
        # α1 = 6/10 / (ϵ + IS1)^2
        # α2 = 1/10 / (ϵ + IS2)^2
    end

    w.w0 = w.α0 / (w.α0 + w.α1 + w.α2)
    w.w1 = w.α1 / (w.α0 + w.α1 + w.α2)
    w.w2 = w.α2 / (w.α0 + w.α1 + w.α2)

    return w.w0, w.w1, w.w2
end

# Lax-Friderichs flux splitting
fplus(u, f, w)  = 1/2 * (f + w.ev * u)
fminus(u, f, w) = 1/2 * (f - w.ev * u)

end

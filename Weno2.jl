# Julia 1.0.1

module Weno

export grid, runge_kutta, weno_dv

function grid(size, min::Float64, max::Float64, ghost)::Tuple{Int, Float64, Vector{Float64}, UnitRange{Int}, Int}
    global gh, nx, dx, x, cr
    gh = ghost
    nx = 2*gh + size
    dx = (max-min)/size
    x  = min - (gh - 1/2)*dx : dx : max + (gh - 1/2)*dx
    cr = 1+gh:nx-gh
    return nx, dx, x, cr, gh
end

# 3rd order TVD Runge-Kutta discretizes
#   du/dt = op(u, x)
# where op is a (nonlinear) operator.
function runge_kutta(u, w, f, eval, dt)
    w.op[cr] = weno_dv(u, w, f, eval)
    @views @. w.u1 = u + dt * w.op
    @views @. w.u2 = 3/4 * u + 1/4 * w.u1 + 1/4 * dt * w.op
    @views @. w.u3 = 1/3 * u + 2/3 * w.u2 + 2/3 * dt * w.op
    return w.u3
end

# 5th order WENO finite-difference scheme
function weno_dv(u, w, f, eval)
    w.fp = fplus(u, f, eval); w.fm = fminus(u, f, eval)
    w.∂f = -1 / dx * (fhat(+1, +0, w.fp, w.fm, w) + fhat(-1, +0, w.fp, w.fm, w) -
                      fhat(+1, -1, w.fp, w.fm, w) - fhat(-1, -1, w.fp, w.fm, w))
    return w.∂f
end

# g =  0 -> j+1/2
# g = -1 -> j-1/2
function fhat(pm::Int, g::Int, fp, fm, w)
    # Computational range, including parameter g
    c = cr .+ g

    # pm: positive/negative flux.
    if pm == 1
        @views @. w.fhat0 = 1/3 * fp[c-2] - 7/6 * fp[c-1] + 11/6 * fp[c]
        @views @. w.fhat1 = -1/6 * fp[c-1] + 5/6 * fp[c] + 1/3 * fp[c+1]
        @views @. w.fhat2 = 1/3 * fp[c] + 5/6 * fp[c+1] - 1/6 * fp[c+2]
        w.w0, w.w1, w.w2 = weights(+1, fp, fm, c, w)
        
    elseif pm == -1
        @views @. w.fhat0 = 1/3 * fm[c+1] + 5/6 * fm[c] - 1/6 * fm[c-1]
        @views @. w.fhat1 = -1/6 * fm[c+2] + 5/6 * fm[c+1] + 1/3 * fm[c]
        @views @. w.fhat2 = 1/3 * fm[c+3] - 7/6 * fm[c+2] + 11/6 * fm[c+1]
        w.w0, w.w1, w.w2 = weights(-1, fp, fm, c, w)
    end

    return @. w.w0 * w.fhat0 + w.w1 * w.fhat1 + w.w2 * w.fhat2
end

function weights(pm, fp, fm, c, w)
    # Small parameter to avoid division by 0
    ϵ = 1e-6

    if pm == 1
        @views @. w.IS0 = 13/12 * (fp[c-2] - 2 * fp[c-1] + fp[c])^2 + 1/4 * (fp[c-2] - 4 * fp[c-1] + 3 * fp[c])^2
        @views @. w.IS1 = 13/12 * (fp[c-1] - 2 * fp[c] + fp[c+1])^2 + 1/4 * (fp[c-1] - fp[c+1])^2
        @views @. w.IS2 = 13/12 * (fp[c] - 2 * fp[c+1] + fp[c+2])^2 + 1/4 * (3 * fp[c] - 4 * fp[c+1] + fp[c+2])^2

        # Yamaleev and Carpenter, 2009
        @views @. w.τ = (fp[c-2] - 4 * fp[c-1] + 6 * fp[c] - 4 * fp[c+1] + fp[c+2])^2
        @views @. w.α0 = 1/10 * (1 + (w.τ / (ϵ + w.IS0))^2)
        @views @. w.α1 = 6/10 * (1 + (w.τ / (ϵ + w.IS1))^2)
        @views @. w.α2 = 3/10 * (1 + (w.τ / (ϵ + w.IS2))^2)

        # Jiang and Shu, 1997
        # @. w.α0 = 1/10 / (ϵ + IS0)^2
        # @. w.α1 = 6/10 / (ϵ + IS1)^2
        # @. w.α2 = 3/10 / (ϵ + IS2)^2 
        
    elseif pm == -1
        @views @. w.IS0 = 13/12 * (fm[c-1] - 2 * fm[c] + fm[c+1])^2 + 1/4 * (fm[c-1] - 4 * fm[c] + 3 * fm[c+1])^2
        @views @. w.IS1 = 13/12 * (fm[c] - 2 * fm[c+1] + fm[c+2])^2 + 1/4 * (fm[c] - fm[c+2])^2
        @views @. w.IS2 = 13/12 * (fm[c+1] - 2 * fm[c+2] + fm[c+3])^2 + 1/4 * (3 * fm[c+1] - 4 * fm[c+2] + fm[c+3])^2

        @views @. w.τ = (fm[c-1] - 4 * fm[c] + 6 * fm[c+1] - 4 * fm[c+2] + fm[c+3])^2  
        @views @. w.α0 = 3/10 * (1 + (w.τ / (ϵ + w.IS0))^2)
        @views @. w.α1 = 6/10 * (1 + (w.τ / (ϵ + w.IS1))^2)
        @views @. w.α2 = 1/10 * (1 + (w.τ / (ϵ + w.IS2))^2)

        # @. w.α0 = 3/10 / (ϵ + IS0)^2
        # @. w.α1 = 6/10 / (ϵ + IS1)^2
        # @. w.α2 = 1/10 / (ϵ + IS2)^2
    end
        
    @views @. w.w0 = w.α0 / (w.α0 + w.α1 + w.α2)
    @views @. w.w1 = w.α1 / (w.α0 + w.α1 + w.α2)
    @views @. w.w2 = w.α2 / (w.α0 + w.α1 + w.α2)

    return w.w0, w.w1, w.w2
end

# Lax-Friderichs flux splitting
fplus(u, f, eval)  = @. 1/2 * (f + eval * u)
fminus(u, f, eval) = @. 1/2 * (f - eval * u)

end





    



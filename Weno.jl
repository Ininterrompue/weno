# Julia 1.0.1

module Weno

export grid, runge_kutta, weno_dv

function grid(size, min::Float64, max::Float64, ghost)
    global gh, nx, dx, x, cr
    gh = ghost
    nx = 2*gh + size
    dx = (max-min)/size
    x  = min - (gh - 1/2)*dx : dx : max + (gh - 1/2)*dx
    cr = 1+gh:nx-gh
    return nx, dx, x, cr
end

# 3rd order TVD Runge-Kutta discretizes
#   du/dt = op(u, x)
# where op is a (nonlinear) operator.
function runge_kutta(u, f, eval, dt)
    op = weno_dv(u, f, eval)
    u1 = u + dt * op
    u2 = 3/4 * u + 1/4 * u1 + 1/4 * dt * op
    u3 = 1/3 * u + 2/3 * u2 + 2/3 * dt * op
    return u3
end

# 5th order WENO finite-difference scheme
function weno_dv(u, f, eval)
    global nx, dx, cr
    fp = fplus(u, f, eval); fm = fminus(u, f, eval)
    ∂f = zeros(nx)
    ∂f[cr] = -1/dx * (fhat(+1, +0, fp, fm) + fhat(-1, +0, fp, fm) -
                      fhat(+1, -1, fp, fm) - fhat(-1, -1, fp, fm))
    return ∂f
end

# g =  0 -> j+1/2
# g = -1 -> j-1/2
function fhat(pm::Int, g::Int, fp, fm)
    global cr
    # Computational range, including parameter g
    c = cr .+ g

    # pm: positive/negative flux.
    if pm == 1
        @views fhat0 = 1/3 * fp[c.-2] - 7/6 * fp[c.-1] + 11/6 * fp[c]
        @views fhat1 = -1/6 * fp[c.-1] + 5/6 * fp[c] + 1/3 * fp[c.+1]
        @views fhat2 = 1/3 * fp[c] + 5/6 * fp[c.+1] - 1/6 * fp[c.+2]
        w0, w1, w2 = weights(+1, fp, fm, c)
        
    elseif pm == -1
        @views fhat0 = 1/3 * fm[c.+1] + 5/6 * fm[c] - 1/6 * fm[c.-1]
        @views fhat1 = -1/6 * fm[c.+2] + 5/6 * fm[c.+1] + 1/3 * fm[c]
        @views fhat2 = 1/3 * fm[c.+3] - 7/6 * fm[c.+2] + 11/6 * fm[c.+1]
        w0, w1, w2 = weights(-1, fp, fm, c)
    end

    return w0 .* fhat0 + w1 .* fhat1 + w2 .* fhat2
end

function weights(pm, fp, fm, c)
    # Small parameter to avoid division by 0
    ϵ = 1e-6

    if pm == 1
        @views IS0 = 13/12 * (fp[c.-2] - 2 * fp[c.-1] + fp[c]).^2 + 1/4 * (fp[c.-2] - 4 * fp[c.-1] + 3 * fp[c]).^2
        @views IS1 = 13/12 * (fp[c.-1] - 2 * fp[c] + fp[c.+1]).^2 + 1/4 * (fp[c.-1] - fp[c.+1]).^2
        @views IS2 = 13/12 * (fp[c] - 2 * fp[c.+1] + fp[c.+2]).^2 + 1/4 * (3 * fp[c] - 4 * fp[c.+1] + fp[c.+2]).^2

        # Yamaleev and Carpenter, 2009
        @views τ = (fp[c.-2] - 4 * fp[c.-1] + 6 * fp[c] - 4 * fp[c.+1] + fp[c.+2]).^2
        α0 = 1/10 * (1 .+ (τ ./ (ϵ .+ IS0)).^2)
        α1 = 6/10 * (1 .+ (τ ./ (ϵ .+ IS1)).^2)
        α2 = 3/10 * (1 .+ (τ ./ (ϵ .+ IS2)).^2)

        # Jiang and Shu, 1997
        # α0 = 1/10 ./ (ϵ .+ IS0).^2
        # α1 = 6/10 ./ (ϵ .+ IS1).^2
        # α2 = 3/10 ./ (ϵ .+ IS2).^2 
        
    elseif pm == -1
        @views IS0 = 13/12 * (fm[c.-1] - 2 * fm[c] + fm[c.+1]).^2 + 1/4 * (fm[c.-1] - 4 * fm[c] + 3 * fm[c.+1]).^2
        @views IS1 = 13/12 * (fm[c] - 2 * fm[c.+1] + fm[c.+2]).^2 + 1/4 * (fm[c] - fm[c.+2]).^2
        @views IS2 = 13/12 * (fm[c.+1] - 2 * fm[c.+2] + fm[c.+3]).^2 + 1/4 * (3 * fm[c.+1] - 4 * fm[c.+2] + fm[c.+3]).^2

        @views τ = (fm[c.-1] - 4 * fm[c] + 6 * fm[c.+1] - 4 * fm[c.+2] + fm[c.+3]).^2  
        α0 = 3/10 * (1 .+ (τ ./ (ϵ .+ IS0)).^2)
        α1 = 6/10 * (1 .+ (τ ./ (ϵ .+ IS1)).^2)
        α2 = 1/10 * (1 .+ (τ ./ (ϵ .+ IS2)).^2)

        # α0 = 3/10 ./ (ϵ .+ IS0).^2
        # α1 = 6/10 ./ (ϵ .+ IS1).^2
        # α2 = 1/10 ./ (ϵ .+ IS2).^2
    end
        
    w0 = α0 ./ (α0 + α1 + α2)
    w1 = α1 ./ (α0 + α1 + α2)
    w2 = α2 ./ (α0 + α1 + α2)

    return w0, w1, w2
end

# Lax-Friderichs flux splitting
fplus(u, f, eval)  = 1/2 * (f + eval * u)
fminus(u, f, eval) = 1/2 * (f - eval * u)

end





    



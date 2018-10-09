# Julia 1.0.1
# Solves the inviscid 1D Burgers' equation
#   ∂u/∂t + ∂f/∂x = 0
#   with periodic boundary conditions.
# By default, f = 1/2 * u^2.

# import Plots

function grid(size, max, ghost)
    global gh, nx, dx, x, cr
    gh = ghost
    nx = 2*gh + size
    dx = max/size
    x = -(gh - 1/2)*dx : dx : max + (gh - 1/2)*dx
    cr = 1+gh:nx-gh
    return nx, dx, x, cr
end

function burgers(u::Vector{Float64}, cfl::Float64, t_max::Float64)
    # Courant condition
    dt::Float64 = dr / 2 * cfl
    t::Float64 = 0; counter::Int = 0

    while t < t_max
        t += dt; counter += 1
        u .= runge_kutta(u, dt)
        # println("$counter   t = $t")

        # Periodic boundary conditions
        u[end-0] = u[6]
        u[end-1] = u[5]
        u[end-2] = u[4]
        u[3] = u[end-3]
        u[2] = u[end-4]
        u[1] = u[end-5]
    end

    # plt = Plots.plot(r[cr], [u₀[cr], u[cr]], linewidth=2, title="Burgers' Equation",
    #                  xaxis="x", yaxis="u(x)", label=["u(x, 0)", "u(x, $t)"])
    # display(plt)
end

# 3rd order TVD Runge-Kutta discretizes
#   du/dt = op(u, x)
# where op is a (nonlinear) operator.
function runge_kutta(u::Vector{Float64}, dt::Float64)::Vector{Float64}
    op = WENO(u)
    u1 = u + dt * op
    u2 = 3/4 * u + 1/4 * u1 + 1/4 * dt * op
    u3 = 1/3 * u + 2/3 * u2 + 2/3 * dt * op
    return u3
end

# 5th order WENO finite-difference scheme
function WENO(u::Vector{Float64})::Vector{Float64}
    fp = fplus(u); fm = fminus(u)
    ∂f = zeros(nr)
    ∂f[cr] = -1/dr * (fhat(+1, +0, fp, fm) + fhat(-1, +0, fp, fm) -
                      fhat(+1, -1, fp, fm) - fhat(-1, -1, fp, fm))
    return ∂f
end

# g =  0 -> j+1/2
# g = -1 -> j-1/2
function fhat(pm::Int, g::Int, fp::Vector{Float64}, fm::Vector{Float64})::Vector{Float64}
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

function weights(pm::Int, fp::Vector{Float64}, fm::Vector{Float64}, c)
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
function fplus(u::Vector{Float64})::Vector{Float64}
    a = maximum(abs.(u))
    return 1/2 * (f(u) + a*u)
end

function fminus(u::Vector{Float64})::Vector{Float64}
    a = maximum(abs.(u))
    return 1/2 * (f(u) - a*u)
end

function f(u::Vector{Float64})::Vector{Float64}
    return 1/2 * u .^2
end



nr, dr, r, cr = grid(256, 10, 3)
u₀ = exp.(-(r .- 5).^2)
# u₀ = sech.(r .- 5)
u = copy(u₀)

burgers(u, 0.3, 1.0)





    



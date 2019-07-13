# Solves the inviscid 1D Burgers' equation with periodic boundary conditions.

include("./Weno.jl")
import .Weno
using Printf
import Plots


function initialize_uf(grpar)
    nx = grpar.nx; x = grpar.x
    u = zeros(nx); f = zeros(nx)
    @. u = exp(-x^2)
    @. f = 1/2 * u^2
    return u, f
end

update_flux!(f, u) = @. f = 1/2 * u^2

CFL_condition(grpar, cfl) = grpar.dx / 2 * cfl

function boundary_conditions!(u)
    u[end-0] = u[6]
    u[end-1] = u[5]
    u[end-2] = u[4]
    u[3] = u[end-3]
    u[2] = u[end-4]
    u[1] = u[end-5]
end

function burgers(; cfl=0.3, t_max=1.0)
    grpar = Weno.grid(size=256, min=-5.0, max=5.0)
    rkpar = Weno.preallocate_rungekutta_parameters(grpar)
    wepar = Weno.preallocate_weno_parameters(grpar)
    u, f = initialize_uf(grpar)
    dt = CFL_condition(grpar, cfl)
    t = 0.0; counter = 0

    while t < t_max
        t += dt; counter += 1
        wepar.ev = maximum(u)
        Weno.time_evolution!(u, f, dt, grpar, rkpar, wepar)
        boundary_conditions!(u)
        update_flux!(f, u)
    end

    @printf("%d iterations. t_max = %2.3f.\n", counter, t)
    x = grpar.x; cr = grpar.cr
    plt = Plots.plot(x[cr], u[cr], title="1D Burgers' Equation", legend=false)
    display(plt)
end

@time burgers(t_max=4.0)

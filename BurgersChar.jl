# Solves the inviscid 1D Burgers' equation
#   ∂u/∂t + ∂f/∂x = 0
#   with periodic boundary conditions.
# By default, f = 1/2 * u^2.

include("./WenoChar.jl")
import .WenoChar
import Printf, BenchmarkTools
import Plots


function initialize_uf(grpar)
    nx = grpar.nx; x = grpar.x
    u = zeros(nx)
    f = zeros(nx)
    for i in 1:nx
        u[i] = exp(-x[i]^2)
        f[i] = 1/2 * u[i]^2
    end
    return u, f
end

function update_flux!(f, u)
    @. f = 1/2 * u^2
end

CFL_condition(grpar, cfl) = grpar.dx / 2 * cfl

function boundary_conditions!(u)
    u[end-0] = u[6]
    u[end-1] = u[5]
    u[end-2] = u[4]
    u[3] = u[end-3]
    u[2] = u[end-4]
    u[1] = u[end-5]
end

function burgers(cfl=0.3, t_max=1.0)
    grpar = WenoChar.grid(256, -5.0, 5.0, 3)
    rkpar = WenoChar.preallocate_rungekutta_parameters(grpar)
    wepar = WenoChar.preallocate_weno_parameters(grpar)
    u, f = initialize_uf(grpar)
    dt = CFL_condition(grpar, cfl)
    t = 0.0; counter = 0

    while t < t_max
        t += dt; counter += 1
        # Printf.@printf("Iteration %d: t = %2.3f\n", counter, t)
        wepar.ev = maximum(u)
        WenoChar.runge_kutta!(u, f, dt, grpar, rkpar, wepar)
        boundary_conditions!(u)

        update_flux!(f, u)
    end

    # Printf.@printf("%d iterations. t_max = %2.3f.\n", counter, t)
    # x = grpar.x; cr = grpar.cr
    # plt = Plots.plot(x[cr], u[cr], title="Burgers' Equation")
    # display(plt)
end

BenchmarkTools.@btime burgers();
# @time burgers()

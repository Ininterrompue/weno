# Solves the inviscid 1D Burgers' equation
#   ∂u/∂t + ∂f/∂x = 0
#   with periodic boundary conditions.
# By default, f = 1/2 * u^2.

include("./WenoChar.jl")
import .WenoChar
import Printf, Plots


function initialize_uf(grpar)
    u = @. exp(-grpar.x^2)
    f = @. 1/2 * u^2
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

function burgers(cfl=0.5, t_max=1.0)
    grpar = WenoChar.grid(64, -5.0, 5.0, 3)
    rkpar = WenoChar.preallocate_rungekutta_parameters(grpar)
    wspar = WenoChar.preallocate_weno_parameters(grpar)
    u, f = initialize_uf(grpar)
    dt = CFL_condition(grpar, cfl)
    t = 0.0; counter = 0

    while t < t_max
        t += dt; counter += 1
        wspar.ev = maximum(u)
        WenoChar.runge_kutta!(u, f, dt, grpar, rkpar, wspar)
        boundary_conditions!(u)
        Printf.@printf("Iteration %d: t = %2.3f\n", counter, t)
        update_flux!(f, u)
    end

    x = grpar.x; cr = grpar.cr
    @show u
    plt = Plots.plot(x[cr], u[cr], linewidth=2, title="Burgers' Equation",
                     xaxis="x", yaxis="u(x)", label="u(x, $t)")
    display(plt)
end

@time burgers()

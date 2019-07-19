# Solves the inviscid 1D Burgers' equation 
#   ∂u/∂t + ∂f/∂x = 0
# with periodic boundary conditions.

include("./Weno.jl")
import .Weno
using Printf
import Plots, BenchmarkTools


function initialize_uf(grpar)
    nx = grpar.nx; x = grpar.x
    u = zeros(nx); f = zeros(nx); f_hat = zeros(nx+1)
    @. u = exp(-x^2)
    @. f = 1/2 * u^2
    return u, f, f_hat
end

initialize_local() = zeros(6), zeros(6)

function update_local!(i, u, f, u_local, f_local)
    for j in 1:6
        u_local[j] = u[i-3+j]
        f_local[j] = f[i-3+j]
    end
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

function plot_system(u, grpar, filename)
    x = grpar.x; cr = grpar.cr_mesh
    plt = Plots.plot(x[cr], u[cr], title="Gaussian Wave", legend=false)
    display(plt)
    Plots.pdf(plt, filename)
end

function burgers(; cfl=0.3, t_max=1.0)
    grpar = Weno.grid(size=512, min=-5.0, max=5.0)
    rkpar = Weno.preallocate_rungekutta_parameters(grpar)
    wepar = Weno.preallocate_weno_parameters(grpar)
    u, f, f_hat = initialize_uf(grpar)
    u_local, f_local = initialize_local()
    dt = CFL_condition(grpar, cfl)
    t = 0.0; counter = 0

    while t < t_max
        t += dt; counter += 1
        wepar.ev = maximum(u)
        for i in grpar.cr_cell
            update_local!(i, u, f, u_local, f_local)
            f_hat[i] = Weno.update_numerical_flux(u_local, f_local, wepar)
        end
        Weno.time_evolution!(u, f_hat, dt, grpar, rkpar)
        boundary_conditions!(u)
        update_flux!(f, u)
    end

    @printf("%d iterations. t_max = %2.3f.\n", counter, t)
    plot_system(u, grpar, "burgers1d_t5_512")
end

# BenchmarkTools.@btime burgers(t_max=4.0);
@time burgers(t_max=5.0)

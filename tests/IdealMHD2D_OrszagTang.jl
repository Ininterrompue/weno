include("../IdealMHD2D.jl")

"""
Orszag-Tang vortex
(x, y) = [0, 2π] × [0, 2π], t_max = 2, 3, 4
"""
function orszagtang!(state, sys)
    γ = sys.γ; Az = state.Az
    Q_prim = state.Q_prim; Q_cons = state.Q_cons
    nx = sys.gridx.nx; ny = sys.gridy.nx

    for j in 1:ny, i in 1:nx
        x = sys.gridx.x[i]
        y = sys.gridy.x[j]
        Q_cons[i, j, 1] = γ^2       # ρ
        Q_prim[i, j, 4] = γ         # P
        Q_prim[i, j, 1] = -sin(y)   # u
        Q_prim[i, j, 2] = +sin(x)   # v
        Q_cons[i, j, 5] = -sin(y)   # Bx
        Q_cons[i, j, 6] = sin(2x)   # By
        Az[i, j] = 1/2 * cos(2x) + cos(y)
    end
end

function idealmhd(; grid_size=64, γ=5/3, cfl=0.4, t_max=0.0, method=:char)
    gridx = grid(size=grid_size, min=0, max=2π)
    gridy = grid(size=grid_size, min=0, max=2π)
    sys = SystemParameters2D(gridx, gridy, 4, 8, γ, 0.0)
    rApar = Weno.preallocate_rungekutta_parameters_2D(gridx, gridy)
    rkpar = Weno.preallocate_rungekutta_parameters_2D(gridx, gridy, sys)
    wepar = Weno.preallocate_weno_parameters()
    state = preallocate_statevectors(sys)
    flux = preallocate_fluxes(sys)
    flxrec = preallocate_fluxreconstruction(sys)
    smooth = preallocate_smoothnessfunctions(sys)

    orszagtang!(state, sys)
    bctype = Periodic()

    primitive_to_conserved!(state, sys)
    boundary_conditions_primitive!(state, sys, bctype)
    boundary_conditions_conserved!(state, sys, bctype)
    boundary_conditions_Az!(state, sys, bctype)

    t = 0.0; counter = 0; t0 = time()
    q = state.Q_local; f = flux.F_local
    t_array = [1.0, 2.0, 3.0, 4.0]; t_counter = 1
    while t < t_max
        update_physical_fluxes!(flux, state, sys)
        dt = CFL_condition(cfl, state, sys, wepar)
        t += dt

        if method == :comp
            for j in gridy.cr_cell, i in gridx.cr_cell
                update_local!(i, j, state.Q_cons, flux.Fx, q, f, sys, :X)
                Weno.update_numerical_fluxes!(i, j, flux.Fx_hat, q, f, sys, wepar, false)
            end
            for j in gridy.cr_cell, i in gridx.cr_cell
                update_local!(i, j, state.Q_cons, flux.Fy, q, f, sys, :Y)
                Weno.update_numerical_fluxes!(i, j, flux.Fy_hat, q, f, sys, wepar, false)
            end
        elseif method == :char
            for j in gridy.cr_cell, i in gridx.cr_cell
                update_xeigenvectors!(i, j, state, flxrec, sys)
                project_to_localspace!(i, j, state, flux, flxrec, sys, :X)
                update_local!(i, j, state.Q_proj, flux.Gx, q, f, sys, :X)
                Weno.update_numerical_fluxes!(i, j, flux.Gx_hat, q, f, sys, wepar, false)
                project_to_realspace!(i, j, flux, flxrec, sys, :X)
            end
            for j in gridy.cr_cell, i in gridx.cr_cell
                update_yeigenvectors!(i, j, state, flxrec, sys)
                project_to_localspace!(i, j, state, flux, flxrec, sys, :Y)
                update_local!(i, j, state.Q_proj, flux.Gy, q, f, sys, :Y)
                Weno.update_numerical_fluxes!(i, j, flux.Gy_hat, q, f, sys, wepar, false)
                project_to_realspace!(i, j, flux, flxrec, sys, :Y)
            end
        elseif method == :ada
            update_smoothnessfunctions!(smooth, state, sys, wepar.ev)
            for j in gridy.cr_cell, i in gridx.cr_cell
                update_local_smoothnessfunctions!(i, j, smooth, wepar, :X)
                Weno.nonlinear_weights_plus!(wepar)
                Weno.nonlinear_weights_minus!(wepar)
                Weno.update_switches!(wepar)
                if wepar.θp > 0.5 && wepar.θm > 0.5
                    update_local!(i, j, state.Q_cons, flux.Fx, q, f, sys, :X)
                    Weno.update_numerical_fluxes!(i, j, flux.Fx_hat, q, f, sys, wepar, true)
                else
                    update_xeigenvectors!(i, j, state, flxrec, sys)
                    project_to_localspace!(i, j, state, flux, flxrec, sys, :X)
                    update_local!(i, j, state.Q_proj, flux.Gx, q, f, sys, :X)
                    Weno.update_numerical_fluxes!(i, j, flux.Gx_hat, q, f, sys, wepar, false)
                    project_to_realspace!(i, j, flux, flxrec, sys, :X)
                end
            end
            for j in gridy.cr_cell, i in gridx.cr_cell
                update_local_smoothnessfunctions!(i, j, smooth, wepar, :Y)
                Weno.nonlinear_weights_plus!(wepar)
                Weno.nonlinear_weights_minus!(wepar)
                Weno.update_switches!(wepar)
                if wepar.θp > 0.5 && wepar.θm > 0.5
                    update_local!(i, j, state.Q_cons, flux.Fy, q, f, sys, :Y)
                    Weno.update_numerical_fluxes!(i, j, flux.Fy_hat, q, f, sys, wepar, true)
                else
                    update_yeigenvectors!(i, j, state, flxrec, sys)
                    project_to_localspace!(i, j, state, flux, flxrec, sys, :Y)
                    update_local!(i, j, state.Q_proj, flux.Gy, q, f, sys, :Y)
                    Weno.update_numerical_fluxes!(i, j, flux.Gy_hat, q, f, sys, wepar, false)
                    project_to_realspace!(i, j, flux, flxrec, sys, :Y)
                end
            end
        end

        for j in gridy.cr_mesh, i in gridx.cr_mesh
            update_local_Az_derivatives!(i, j, state, sys, wepar, :X)
            update_numerical_Az_derivatives!(i, j, state, flux, wepar, :X)
        end
        for j in gridy.cr_mesh, i in gridx.cr_mesh
            update_local_Az_derivatives!(i, j, state, sys, wepar, :Y)
            update_numerical_Az_derivatives!(i, j, state, flux, wepar, :Y)
        end

        time_evolution_Az!(state, flux, sys, dt, rApar)
        boundary_conditions_Az!(state, sys, bctype)

        time_evolution!(state, flux, sys, dt, rkpar)
        correct_magneticfield!(state, sys)
        boundary_conditions_conserved!(state, sys, bctype)

        conserved_to_primitive!(state, sys)
        boundary_conditions_primitive!(state, sys, bctype)

        counter += 1
        if counter % 100 == 0
            @printf("Iteration %d: t = %2.3f, dt = %2.3e, v_max = %6.5f, Elapsed time = %3.3f\n", 
                counter, t, dt, wepar.ev, time() - t0)
        end
        # if t > t_array[t_counter]
        #     plot_system(state.Q_cons[:, :, 1], sys, "Rho", "rho_$(grid_size)_t$(t_counter)_" * string(method))
        #     plot_system(state.Q_prim[:, :, 4], sys, "P", "P_$(grid_size)_t$(t_counter)_" * string(method))
        #     plot_system(state.Az, sys, "Az", "Az_$(grid_size)_t$(t_counter)_" * string(method))
        #     plot_system(state.Q_cons[:, :, 5], sys, "Bx", "Bx_$(grid_size)_t$(t_counter)_" * string(method))
        #     plot_system(state.Q_cons[:, :, 6], sys, "By", "By_$(grid_size)_t$(t_counter)_" * string(method))
        
        #     T = calculate_temperature(state, sys)
        #     plot_system(T, sys, "T", "T_$(grid_size)_t$(t_counter)_" * string(method))

        #     divB = calculate_divergence(state, sys)
        #     plot_system(divB, sys, "divB", "divB_$(grid_size)_t$(t_counter)_" * string(method))

        #     t_counter += 1
        # end
    end
    @printf("%d iterations. t_max = %2.3f. Elapsed time = %3.3f\n", 
        counter, t, time() - t0)
    
    plot_system(state.Q_cons[:, :, 1], sys, "Rho", "rho_$(grid_size)_t$(t_counter)_" * string(method))
    plot_system(state.Q_prim[:, :, 4], sys, "P", "P_$(grid_size)_t$(t_counter)_" * string(method))
    plot_system(state.Q_prim[:, :, 1], sys, "u", "u_$(grid_size)_t$(t_counter)_" * string(method))
    plot_system(state.Q_prim[:, :, 2], sys, "v", "v_$(grid_size)_t$(t_counter)_" * string(method))
    plot_system(state.Az, sys, "Az", "Az_$(grid_size)_t$(t_counter)_" * string(method))
    plot_system(state.Q_cons[:, :, 5], sys, "Bx", "Bx_$(grid_size)_t$(t_counter)_" * string(method))
    plot_system(state.Q_cons[:, :, 6], sys, "By", "By_$(grid_size)_t$(t_counter)_" * string(method))

    T = calculate_temperature(state, sys)
    plot_system(T, sys, "T", "T_$(grid_size)_t$(t_counter)_" * string(method))
end

@time idealmhd(grid_size=128, t_max=2.0, method=:ada)
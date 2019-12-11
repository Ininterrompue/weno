include("../IdealMHD2D.jl")

struct SemiPeriodic <: BoundaryCondition end

"""
MHD wave test
(x, y) = [-1, 1] × [-1, 1], t_max = 0.5
"""
function alfvenwave!(state, sys)
    γ = sys.γ; A_A = sys.A_A; Az = state.Az
    Q_prim = state.Q_prim; Q_cons = state.Q_cons
    nx = sys.gridx.nx; ny = sys.gridy.nx

    for j in 1:ny, i in 1:nx
        x = sys.gridx.x[i]
        y = sys.gridy.x[j]
        Q_cons[i, j, 1] = 1.0   # ρ
        Q_prim[i, j, 4] = 1/γ   # P
        Q_prim[i, j, 2] = A_A * cos(2π*x)       # v
        Q_cons[i, j, 5] = 1.0   # Bx
        Q_cons[i, j, 6] = -A_A * cos(2π*x)      # By
        Az[i, j] = y + 1/2π * A_A * sin(2π*x)   # Az
    end
end

"""
MHD wave test 2: circularly polarized Alfven waves
"""
function alfven2waves!(state, sys)
    γ = sys.γ; A_A = sys.A_A; Az = state.Az
    Q_prim = state.Q_prim; Q_cons = state.Q_cons
    nx = sys.gridx.nx; ny = sys.gridy.nx

    for j in 1:ny, i in 1:nx
        x = sys.gridx.x[i]
        y = sys.gridy.x[j]
        Q_cons[i, j, 1] = 1.0   # ρ
        Q_prim[i, j, 4] = 1/γ   # P
        Q_prim[i, j, 2] = A_A * ( cos(2π*x) - cos(π*x) )       # v
        Q_cons[i, j, 5] = 1.0   # Bx
        Q_cons[i, j, 6] = -A_A * ( cos(2π*x) - cos(π*x) )     # By
        Az[i, j] = y + 1/2π * A_A * ( sin(2π*x) - sin(π*x) )   # Az
    end 
end

function alfvenwave_error(sys, state, t)
    nx = sys.gridx.nx; ny = sys.gridy.nx
    cry = sys.gridy.cr_mesh; crx = sys.gridx.cr_mesh
    By = zeros(nx, ny)
    error = 0.0
    for j in 1:ny, i in 1:nx
        x = sys.gridx.x[i]
        By[i, j] = -sys.A_A * cos(2π*(x-t))
    end
    for j in cry, i in crx
        error += (state.Q_cons[i, j, 6] - By[i, j])^2 / (nx * ny)
    end
    return error
end

function alfvenwave_difference(sys, state, t)
    A_A = sys.A_A
    crx = sys.gridx.cr_mesh; cry = sys.gridy.cr_mesh
    By_analytical = zeros(sys.gridx.nx)
    for i in crx
        x = sys.gridx.x[i]
        By_analytical[i] = -A_A * cos(2π*(x - t))
    end
    By_numerical = state.Q_cons[:, sys.gridx.nx ÷ 2, 6]
    A_difference = maximum(By_numerical[crx] - By_analytical[crx])
end

boundary_conditions_conserved!(state, sys, bctype::SemiPeriodic) = 
    boundary_conditions_conserved!(state, sys, Periodic()) 

boundary_conditions_primitive!(state, sys, bctype::SemiPeriodic) = 
    boundary_conditions_primitive!(state, sys, Periodic())

function boundary_conditions_Az!(state, sys, bctype::SemiPeriodic)
    Az = state.Az
    nx = sys.gridx.nx; ny = sys.gridy.nx; y = sys.gridy.x
    for i in 1:nx
        Az[i, end-0] = Az[i, 6] + (y[end-0] - y[6])
        Az[i, end-1] = Az[i, 5] + (y[end-1] - y[5])
        Az[i, end-2] = Az[i, 4] + (y[end-2] - y[4])
        Az[i, 3] = Az[i, end-3] + (y[3] - y[end-3])
        Az[i, 2] = Az[i, end-4] + (y[2] - y[end-4])
        Az[i, 1] = Az[i, end-5] + (y[1] - y[end-5])
    end
    for j in 1:ny
        Az[end-0, j] = Az[6, j]
        Az[end-1, j] = Az[5, j]
        Az[end-2, j] = Az[4, j]
        Az[3, j] = Az[end-3, j]
        Az[2, j] = Az[end-4, j]
        Az[1, j] = Az[end-5, j]
    end
end

function plot_alfvenwaves(sys, state, t_max)
    A_A = sys.A_A
    crx = sys.gridx.cr_mesh; cry = sys.gridy.cr_mesh
    By_analytical = zeros(sys.gridx.nx)
    By_0 = zeros(sys.gridx.nx)
    for i in crx
        x = sys.gridx.x[i]
        By_0[i] = -A_A * cos(2π*x)
        By_analytical[i] = -A_A * cos(2π*(x - t_max))
    end
    By_numerical = state.Q_cons[:, sys.gridx.nx ÷ 2, 6]
    By_difference = By_numerical - By_analytical
    plt = Plots.plot(sys.gridx.x[crx], By_difference[crx], markershape=:circle, label="ByD")
    Plots.plot!(sys.gridx.x[crx], By_analytical[crx], markershape=:circle, label="ByA")
    Plots.plot!(sys.gridx.x[crx], By_numerical[crx], markershape=:circle, label="ByN")
    display(plt)
    # Plots.pdf(plt, filename)
end


function idealmhd(; grid_size=64, γ=5/3, A_A=1e-2, cfl=0.4, t_max=0.0)
    gridx = grid(size=grid_size, min=-1.0, max=1.0)
    gridy = grid(size=grid_size, min=-1.0, max=1.0)
    sys = SystemParameters2D(gridx, gridy, 4, 8, γ, A_A)
    rApar = Weno.preallocate_rungekutta_parameters_2D(gridx, gridy)
    rkpar = Weno.preallocate_rungekutta_parameters_2D(gridx, gridy, sys)
    wepar = Weno.preallocate_weno_parameters()
    state = preallocate_statevectors(sys)
    flux = preallocate_fluxes(sys)
    flxrec = preallocate_fluxreconstruction(sys)
    smooth = preallocate_smoothnessfunctions(sys)

    t_range = 0.0:0.01:1.0
    A_t = zeros(length(t_range))
    divU = zeros(length(t_range)+1)

    # alfven2waves!(state, sys)
    alfvenwave!(state, sys)
    bctype = SemiPeriodic()

    primitive_to_conserved!(state, sys)
    boundary_conditions_primitive!(state, sys, bctype)
    boundary_conditions_conserved!(state, sys, bctype)
    boundary_conditions_Az!(state, sys, bctype)

    t = 0.0; counter = 0; t0 = time()
    q = state.Q_local; f = flux.F_local

    # divU[1] = calculate_divergenceU(state, sys)
    t_counter = 1
    while t < t_max
        update_physical_fluxes!(flux, state, sys)
        dt = CFL_condition(cfl, state, sys, wepar)
        t += dt

        # Characteristic-wise reconstruction
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
        # if t > t_range[t_counter]
        #     @printf("Iteration %d: t = %2.3f, dt = %2.3e, v_max = %6.5f, Elapsed time = %3.3f\n", 
        #         t_counter, t, dt, wepar.ev, time() - t0)
        #     # A_t[t_counter] = alfvenwave_difference(sys, state, t)
        #     # divU[t_counter] = calculate_divergenceU(state, sys)
        #     t_counter += 1
        # end
    end

    @printf("%d grid size. %d iterations. t_max = %2.3f. Elapsed time = %3.3f.\n", 
        grid_size, counter, t, time() - t0)
    # plot_alfvenwaves(sys, state, t_max)

    # divU = calculate_divergenceU(state, sys)
    # plot_system(divU, sys, "divU", "")
    
    # plot_system(state.Q_cons[:, :, 1], sys, "Rho", "")
    # plot_system(state.Q_prim[:, :, 4], sys, "P", "")
    # plot_system(state.Q_prim[:, :, 1], sys, "u", "")
    # plot_system(state.Q_prim[:, :, 2], sys, "v", "")
    # plot_system(state.Az, sys, "Az", "")
    # plot_system(state.Q_cons[:, :, 5], sys, "Bx", "")
    # plot_system(state.Q_cons[:, :, 6], sys, "By", "")

    return alfvenwave_error(sys, state, t_max)
    # return alfvenwave_difference(sys, state, t_max)
    # return A_t
    # return divU
end

function alfvenwave_test()
    grid_size_range = 8:8:64
    A_A_exponents = 1:5
    error_array = zeros(length(grid_size_range), length(A_A_exponents))

    for j in 1:length(A_A_exponents), i in 1:length(grid_size_range)
        amplitude = 10.0^(-A_A_exponents[j])
        size = grid_size_range[i]
        error_array[i, j] = idealmhd(grid_size=size, γ=5/3, A_A=amplitude, cfl=0.4, t_max=0.25)
    end

    plt = Plots.plot(grid_size_range, error_array, legend=false, 
        markershape=:circle, yscale=:log10, title="Alfven wave convergence",
        xlabel="Grid size", ylabel="Average grid error")
    display(plt)
    Plots.pdf(plt, "test_linearwave_convergence")
end

function alfvenwave_amplitudescaling()
    A_A_exponents = -2:0.5:10
    A_difference = zeros(length(A_A_exponents))

    for i in 1:length(A_A_exponents)
        amplitude = 2.0^(-A_A_exponents[i])
        A_difference[i] = idealmhd(grid_size=64, γ=5/3, A_A=amplitude, cfl=0.4, t_max=0.25)
    end
    @show A_difference

    plt = Plots.plot(2.0 .^(-1 * A_A_exponents), A_difference, legend=false, xscale=:log2, yscale=:log2,
        markershape=:circle, title="Amplitude of ByN - ByA")
    display(plt)
    Plots.pdf(plt, "difference_amplitude_scaling")
end

function alfvenwave_amplitudevst()
    amplitude = 0.2
    t_range = 0.0:0.01:1.0

    A_t = idealmhd(grid_size=64, γ=5/3, A_A=amplitude, cfl=0.4, t_max=1.0)

    plt = Plots.plot(t_range[2:end], A_t[2:end], legend=false, # yscale=:log2,
        title="Amplitude of ByN - ByA", xlabel="t")
    display(plt)
    # Plots.pdf(plt, "difference_amplitude_vst_2e-1")
end

function alfvenwave_compression()
    amplitude = 1e-2
    t_range = 0.0:0.01:1.0

    divU = idealmhd(grid_size=32, γ=5/3, A_A=amplitude, cfl=0.4, t_max=1.0)

    plt = Plots.plot(t_range, divU, legend=false,
        title="Amplitude of abs(divU)", xlabel="t")
    display(plt)
    # Plots.pdf(plt, "amplitude_divU_1e-2")
end

# alfvenwave_amplitudescaling()
alfvenwave_test()
# alfvenwave_amplitudevst()
# alfvenwave_compression()
# idealmhd(grid_size=64, γ=5/3, A_A=0.5, cfl=0.4, t_max=0.5)
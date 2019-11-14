# Default: T = Float64, S = Int64

struct GridParameters{T, S}
    nx::S                      # number of points
    dx::T                      # grid spacing
    x::StepRangeLen{T, T, T}   # linspace array
    cr_mesh::UnitRange{S}      # computational range of mesh points
    cr_cell::UnitRange{S}      # computational range of cell boundaries
    ghost::S                   # number of ghost points
end

struct SystemParameters1D{T, S}
    gridx::GridParameters{T, S}   # X-grid
    nprim::S                      # number of nonconserved variables
    ncons::S                      # number of conserved variables
    γ::T                          # adiabatic index
end

struct SystemParameters2D{T, S}
    gridx::GridParameters{T, S}
    gridy::GridParameters{T, S}
    nprim::S
    ncons::S
    γ::T
    A_A::T
end

function grid(; size=32, min=-1.0, max=1.0, ghost=3)
    nx = 2*ghost + size
    dx = (max-min)/(size-1)
    x::StepRangeLen{Float64, Float64, Float64} = 
        min - ghost*dx : dx : max + ghost*dx
    cr_mesh = ghost+1:nx-ghost
    cr_cell = ghost:nx-ghost
    return GridParameters(nx, dx, x, cr_mesh, cr_cell, ghost)
end


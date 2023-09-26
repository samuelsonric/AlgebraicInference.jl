```@meta
EditURL = "../../literate/pixel.jl"
```

# Pixel Arrays

````@example pixel
using AlgebraicInference
using Catlab
using UnicodePlots

using Catlab.CategoricalAlgebra.FinRelations: BoolRig
````

Spivak, Dobson, Kumari, and Wu, *Pixel Arrays*: A fast and elementary method for solving nonlinear systems.

A pixel array is an array with entries in the boolean semiring.

````@example pixel
const PixelArray{N} = Array{BoolRig, N}

function PixelArray(f::Function, xdim::NamedTuple, ydim::NamedTuple, tol::Real)
    rows = xdim.resolution
    cols = ydim.resolution

    result = PixelArray{2}(undef, rows, cols)

    xstep = (xdim.upper - xdim.lower) / xdim.resolution
    ystep = (ydim.upper - ydim.lower) / ydim.resolution

    for i in 1:rows, j in 1:cols
        xval = xdim.lower + (i - 0.5) * xstep
        yval = ydim.lower + (j - 0.5) * ystep

        try
            result[i, j] = -tol < f(xval, yval) < tol
        catch
            result[i, j] = false
        end
    end

    result
end
````

For plotting.

````@example pixel
function Base.isless(x::Int, y::BoolRig)
    y == true ? x < 1 : x < 0
end;
nothing #hide
````

Example 2.4.1

````@example pixel
function f₁(x::Real, z::Real)
    0 - (cos(log(z^2 + x / 10^3)) − x + 1 / z / 10^5)
end

function f₂(w::Real, y::Real)
    2 - (cosh(w + y / 10^3) + y + w / 10^4)
end

function f₃(x::Real, y::Real)
    1 - (tan(x + y) / (x - 2) / (x + 3) / y^2)
end

w = x = y = z = (lower = -3, upper = 3, resolution = 125)

tol = .2

A₁ = PixelArray(f₁, x, z, tol)
A₂ = PixelArray(f₂, w, y, tol)
A₃ = PixelArray(f₃, x, y, tol);
nothing #hide
````

f₁(x, z)

````@example pixel
spy(A₁)
````

f₂(w, y)

````@example pixel
spy(A₂)
````

f₃(x, y)

````@example pixel
spy(A₃)
````

````@example pixel
diagram = @relation (w, z) where (w::n, x::n, y::n, z::n) begin
    f₁(x, z)
    f₂(w, y)
    f₃(x, y)
end

to_graphviz(diagram; box_labels=:name, junction_labels=:variable)
````

````@example pixel
hom_map = Dict{Symbol, PixelArray}(
    :f₁ => A₁,
    :f₂ => A₂,
    :f₃ => A₃)

ob_map = Dict(
    :n => 125)

problem = InferenceProblem(diagram, hom_map, ob_map)

A = solve(problem)

spy(A)
````


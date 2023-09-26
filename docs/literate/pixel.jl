# # Pixel Arrays
using AlgebraicInference
using Catlab
using UnicodePlots

using Catlab.CategoricalAlgebra.FinRelations: BoolRig
# A pixel array is an array with entries in the boolean semiring. The function defined below
# constructs a pixel array by discretizing a relation of the form
# ```math
# xRy \iff f(x, y) = 0.
# ```
#
# References:
# - Spivak, David I. et al. "Pixel Arrays: A fast and elementary method for solving
#   nonlinear systems." *arXiv: Numerical Analysis* (2016): n. pag.
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
end;
# For plotting:
function Base.isless(x::Int, y::BoolRig)
    y == true ? x < 1 : x < 0
end;
# Example 2.4.1 in Spivak et. al.
# ```math
# \begin{align*}
# & \text{Solve relations:}      && \cos(\ln(z^2 + 10^{-3}x)) - x + 10^{-5}z^{-1} = 0 && \text{(Equation 1)} \\
# &                              && \cosh(w + 10^{-3}y) + y + 10^{-4}w = 2            && \text{(Equation 2)} \\
# &                              && \tan(x + y)(x - 2)^{-1}(x + 3)^{-1}y^{-2} = 1     && \text{(Equation 3)} \\
# & \text{Range and resolution:} && w, x, y, z \in [-3, 3)@125                        &&                     \\
# & \text{Expose variables:}     && (w, z)                                            &&                     \\
# & \end{align*}
# ```
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

R₁ = PixelArray(f₁, x, z, tol)
R₂ = PixelArray(f₂, w, y, tol)
R₃ = PixelArray(f₃, x, y, tol);
#
spy(R₁; title="Equation 1", xlabel="x", ylabel="z")
#
spy(R₂; title="Equation 2", xlabel="w", ylabel="y")
#
spy(R₃; title="Equation 3", xlabel="x", ylabel="y")
#
diagram = @relation (w, z) where (w::n, x::n, y::n, z::n) begin
    R₁(x, z)
    R₂(w, y)
    R₃(x, y)
end

to_graphviz(diagram; box_labels=:name, junction_labels=:variable)
#
hom_map = Dict{Symbol, PixelArray}(
    :R₁ => R₁,
    :R₂ => R₂,
    :R₃ => R₃)

ob_map = Dict(
    :n => 125)

problem = InferenceProblem(diagram, hom_map, ob_map)

R = solve(problem)

spy(R; title="Solution", xlabel="w", ylabel="z")
#

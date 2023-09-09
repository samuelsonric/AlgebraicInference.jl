# Compute
# X' * A * X,
# where A is positive semidefinite.
function Xt_A_X(A::AbstractMatrix, X::AbstractMatrix)
    Symmetric(X' * A * X)
end


# Get the index of the minimum value of v. Halts early if
# vᵢ ≤ bound.
function ssargmin(v::AbstractVector, bound)
    ssargmin(identity, v, bound)
end


# Get the index of the minimum value of f.v. Halts early if
# f(vᵢ) ≤ bound.
function ssargmin(f::Function, v::AbstractVector, bound)
    imin = 1
    ymin = f(v[1])

    for (i, x) in enumerate(v)
        y = f(x)

        if y < ymin
            imin = i
            ymin = y
        end

        if y <= bound
            break
        end
    end

    imin
end

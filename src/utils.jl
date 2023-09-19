# Compute
# X' * A * X,
# where A is positive semidefinite.
function Xt_A_X(A, X)
    Symmetric(X' * A * X)
end


# Split a square matrix M into blocks:
# A = [ M₁₁ M₁₂
#       M₂₁ M₂₂ ].
function blocks(M::AbstractMatrix, i₁::AbstractVector, i₂::AbstractVector)
    M₁₁ = M[i₁, i₁]
    M₂₂ = M[i₂, i₂]
    M₁₂ = M[i₁, i₂]
    M₂₁ = M[i₂, i₁]

    M₁₁, M₂₂, M₁₂, M₂₁
end


function blocks(M::Diagonal, i₁::AbstractVector, i₂::AbstractVector)
    v₁, v₂ = blocks(diag(M), i₁, i₂)

    n₁ = length(v₁)
    n₂ = length(v₂)

    M₁₁ = Diagonal(v₁)
    M₂₂ = Diagonal(v₂)
    M₁₂ = Zeros(n₁, n₂)
    M₂₁ = Zeros(n₂, n₁)

    M₁₁, M₂₂, M₁₂, M₂₁
end


# Split a vector v into blocks:
# v = [ v₁
#       v₂ ]
function blocks(v::AbstractVector, i₁::AbstractVector, i₂::AbstractVector)
    v₁ = v[i₁]
    v₂ = v[i₂]

    v₁, v₂
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

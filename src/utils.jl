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


# Add an element x to a sorted set v.
function insertsorted!(v::AbstractVector, x)
    i = searchsortedfirst(v, x)

    if i > length(v) || v[i] != x
        insert!(v, i, x)
        true
    else
        false
    end
end


# Delete an element x from a sorted set v.
function deletesorted!(v::AbstractVector, x)
    i = searchsortedfirst(v, x)

    if i <= length(v) && v[i] == x
        deleteat!(v, i)
        true
    else
        false
    end
end


# Like argmin, but halts early if
# vᵢ ≤ bound
function _argmin(v::AbstractVector, bound)
    _argmin(i -> v[i], eachindex(v), bound)
end


# Like argmin, but halts early if
# f(vᵢ) ≤ bound
function _argmin(f::Function, v::AbstractVector, bound)
    xmin = first(v)
    ymin = f(xmin)

    for x in v
        y = f(x)

        if y < ymin
            xmin = x
            ymin = y
        end

        if y <= bound
            break
        end
    end

    xmin
end


# Determine whether the diagram is an inference problem.
function isvalid(diagram::AbstractUWD)
    B = Tuple{Int, Int}[]
    b = Int[]

    for i in ports(diagram)
        f = box(diagram, i)
        v = junction(diagram, i)::Int

        if !insertsorted!(B, (f, v))
            return false
        end

        insertsorted!(b, v)
    end

    if length(b) != njunctions(diagram)
        return false
    end

    empty!(b)

    for i in ports(diagram; outer=true)
        v = junction(diagram, i; outer=true)::Int

        if !insertsorted!(b, v)
            return false
        end
    end

    true
end

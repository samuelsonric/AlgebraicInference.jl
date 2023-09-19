# A type for solving linear problems of the form
# [ A B'] [ x ] = [ f ]
# [ B 0 ] [ y ]   [ g ]
# where A is positive semidefinite.
struct KKT{T₁, T₂, T₃, T₄, T₅, T₆}
    A::T₁
    U₁::T₂
    V₁::T₃
    V₂::T₄
    cache₁::T₅
    cache₂::T₆
end


# Construct a KKT matrix of the form
# [ A B']
# [ B 0 ]
# where A is positive semidefinite.
function KKT(A::AbstractMatrix, B::AbstractMatrix; atol::Real=1e-8)
    U, S, V = svd(B; full=true)
    n = sum(S .> atol)

    S₁ = S[1:n]
    U₁ = U[:, 1:n]
    V₁ = V[:, 1:n]; V₂ = V[:, n + 1:end]

    KKT(A, S₁, U₁, V₁, V₂)
end


function KKT(A::AbstractMatrix, B::ZerosMatrix; atol::Real=1e-8)
    m, n = size(B)

    S₁ = Zeros(0)
    U₁ = Zeros(m, 0)
    V₁ = Zeros(n, 0); V₂ = Eye(n)

    KKT(A, S₁, U₁, V₁, V₂)
end


function KKT(A::AbstractMatrix, B::Diagonal; atol::Real=1e-8)
    S = diag(B); U = V = Eye(B)
    i = S .> atol

    S₁ = S[i]
    U₁ = U[:, i]
    V₁ = V[:, i]; V₂ = V[:, .!i]

    KKT(A, S₁, U₁, V₁, V₂)
end


# Construct a KKT matrix of the form
# [ A B']
# [ B 0 ]
# where A is positive semidefinite, and B is given by its singular-value decomposition:
# B = [ U₁ U₂ ] [ S₁ 0 ] [ V₁' ]
#               [ 0  0 ] [ V₂' ]
function KKT(
    A::AbstractMatrix,
    S₁::AbstractVector,
    U₁::AbstractMatrix,
    V₁::AbstractMatrix,
    V₂::AbstractMatrix)

    A₁ = Diagonal(S₁)
    A₂ = Xt_A_X(A, V₂)

    b₁ = zeros(size(A₁, 1))
    b₂ = zeros(size(A₂, 1))

    alg₁ = DiagonalFactorization()
    alg₂ = CholeskyFactorization()

    cache₁ = init(LinearProblem(A₁, b₁), alg₁)
    cache₂ = init(LinearProblem(A₂, b₂), alg₂)

    KKT(A, U₁, V₁, V₂, cache₁, cache₂)
end


# Solve for x:
# [ A B'] [ x ] = [ f ]
# [ B 0 ] [ y ]   [ g ]
# where A is positive semidefinite.
function CommonSolve.solve!(K::KKT, f::AbstractVector, g::AbstractVector)
    solve!(K, convert(Vector{Float64}, f), convert(Vector{Float64}, g))
end


function CommonSolve.solve!(K::KKT, f::AbstractVector, g::ZerosVector)
    solve!(K, convert(Vector{Float64}, f), g)
end


function CommonSolve.solve!(K::KKT, f::Vector{Float64}, g::Vector{Float64})
    K.cache₁.b = K.U₁' * g
    x₁ = K.V₁ * solve!(K.cache₁)

    K.cache₂.b = K.V₂' * (f - K.A * x₁)
    x₂ = K.V₂ * solve!(K.cache₂)

    x₁ + x₂
end


function CommonSolve.solve!(K::KKT, f::Vector{Float64}, g::ZerosVector)
    K.cache₂.b = K.V₂' * f
    K.V₂ * solve!(K.cache₂)
end


# Solve for X:
# [ A B'] [ X ] = [ F ]
# [ B 0 ] [ Y ]   [ G ]
# where A is positive semidefinite.
function CommonSolve.solve!(K::KKT, F::AbstractMatrix, G::AbstractMatrix)
    m, n = size(F)
    X = zeros(m, n)

    for i in 1:n
        X[:, i] = solve!(K, F[:, i], G[:, i])
    end

    X
end

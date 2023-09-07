# A KKT matrix of the form
# [ A B']
# [ B 0 ]
# where A is positive semidefinite.
struct KKT{T₁, T₂, T₃, T₄, T₅}
    A::T₁
    B::T₂
    U::T₃
    cache₁::T₄
    cache₂::T₅
end


# Construct a KKT matrix of the form
# [ A B']
# [ B 0 ]
# where A is positive semidefinite.
function KKT(A, B, alg=KrylovJL_MINRES(); atol=1e-8)
    U = nullspace(B; atol)'
    A₁ = B * B'
    A₂ = U * A * U'
    b₁ = zeros(size(B, 1))
    b₂ = zeros(size(U, 1))
    cache₁ = init(LinearProblem(A₁, b₁), alg)
    cache₂ = init(LinearProblem(A₂, b₂), alg)
    KKT(A, B, U, cache₁, cache₂)
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
    K.cache₁.b = g
    x₁ = K.B' * solve!(K.cache₁)
    K.cache₂.b = K.U * (f - K.A * x₁)
    x₂ = K.U' * solve!(K.cache₂)
    x₁ + x₂
end


function CommonSolve.solve!(K::KKT, f::Vector{Float64}, g::ZerosVector)
    K.cache₂.b = K.U * f
    K.U' * solve!(K.cache₂)
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

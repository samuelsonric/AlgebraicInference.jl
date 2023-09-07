# Compute
# X' * A * X,
# where A is positive semidefinite.
function Xt_A_X(A::AbstractMatrix, X::AbstractMatrix)
    Symmetric(X' * A * X)
end



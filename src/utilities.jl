# Compute the pushout of the diagram
#    L     R
# m --> k <-- n
function pushout(L::AbstractMatrix, R::AbstractMatrix)
    n = size(L, 1)
    P = nullspace([L' -R'])'
    ιL = P[:, 1:n]
    ιR = P[:, n+1:end]
    ιL, ιR
end

# Solve for y:
# [ A B'] [ x ] = [ f ]
# [ B 0 ] [ y ]   [ 0 ]
# where A is positive semidefinite.
function solve_mean(A::AbstractMatrix, B::AbstractMatrix, f::AbstractVector)
    V = [A B'
         B 0I]
    n = size(A, 1)
    M = pinv(V)[n+1:end, 1:n]
    M * f
end

# Solve for Y:
# [ A B'] [ X Z'] [ A B'] = [ A 0 ]
# [ B 0 ] [ Z Y ] [ B 0 ]   [ 0 0 ]
# where A is positive semidefinite.
function solve_cov(A::AbstractMatrix, B::AbstractMatrix)
    V = [A B'
         B 0I]
    n = size(A, 1)
    M = pinv(V)[n+1:end, 1:n]
    M * A * M'
end

# Given a directed join tree
#   (V, E),
# get the child of vertex i < |V|.
function ch(V::Integer, E::AbstractSet, i::Integer)
    @assert i < V
    for j in i + 1:V
        if Set([i, j]) in E
            return j
        end
    end
    error()
end

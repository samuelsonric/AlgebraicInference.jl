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
# [ AA' B'] [ x ] = [ f ]
# [ B   0 ] [ y ]   [ 0 ]
function solve1(A::AbstractMatrix, B::AbstractMatrix, f::AbstractVector)
    V = [ A*A' B'
           B    0I ]
    n = size(A, 1)
    pinv(V)[n+1:end, 1:n] * f
end

# Solve for Y:
# [ AA' B'] [ X ]  = [ A ]
# [ B   0 ] [ Y ]    [ 0 ]
function solve2(A::AbstractMatrix, B::AbstractMatrix)
    V = [ A*A' B'
         B    0I ]
    n = size(A, 1)
    pinv(V)[n+1:end, 1:n] * A
end

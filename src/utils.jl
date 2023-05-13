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

function message_to_parent(factors::AbstractVector{<:Valuation{T₁}},
                           domains::AbstractVector{T₂},
                           tree::Node{Int}) where {T₁ <: Variable, T₂ <: AbstractSet{T₁}}
    @assert isdefined(tree, :parent)
    factor = factors[tree.id]
    for subtree in tree.children
        message = message_to_parent(factors, domains, subtree)
        factor = combine(factor, message)
    end
    message = project(factor, domain(factor) ∩ domains[tree.parent.id])
    message
end

function message_to_parent!(mailboxes::AbstractDict{Tuple{Int, Int}, Valuation{T₁}},
                            factors::AbstractVector{<:Valuation{T₁}},
                            domains::AbstractVector{T₂},
                            tree::Node{Int}) where {T₁ <: Variable, T₂ <: AbstractSet{T₁}}
    @assert isdefined(tree, :parent)
    get(mailboxes, (tree.id, tree.parent.id)) do
        factor = factors[tree.id]
        for subtree in tree.children
            message = message_to_parent!(factors, domains, subtree, mailboxes)
            factor = combine(factor, message)
        end
        message = project(factor, domain(factor) ∩ domains[tree.parent.id])
        mailboxes[tree.id, tree.parent.id] = message
        message
    end
end

function message_from_parent!(mailboxes::AbstractDict{Tuple{Int, Int}, Valuation{T₁}},
                              factors::AbstractVector{<:Valuation{T₁}},
                              domains::AbstractVector{T₂},
                              tree::Node{Int}) where {T₁ <: Variable, T₂ <: AbstractSet{T₁}}
    get(mailboxes, (tree.parent.id, tree.id)) do
        factor = factors[tree.parent.id]
        for subtree in tree.parent.children
            message = message_to_parent!(mailboxes, factors, domains, subtree)
            factor = combine(factor, message)
        end
        if isdefined(tree.parent, :parent)
            message = message_from_parent!(mailboxes, factors, domains, tree.parent)
            factor = combine(factor, message)
        end
        message = project(factor, domain(factor) ∩ domains[tree.id])
        mailboxes[tree.parent.id, tree.id] = message
        message
    end
end

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

# Compute the message
# μ i -> pa(i)
function message_to_parent(node::Architecture)
    @assert !isroot(node)
    if isnothing(node.message_to_parent)
        factor = node.factor
        for child in node.children
            factor = combine(factor, message_to_parent(child))
        end
        project(factor, domain(factor) ∩ node.parent.domain)
    else
        node.message_to_parent
    end
end

# Compute the message
# μ pa(i) -> i
function message_from_parent(node::Architecture)
    @assert !isroot(node)
    if isnothing(node.message_from_parent)
        factor = node.factor
        for sibling in node.parent.children
            if node.id != sibling.id
                factor = combine(factor, message_to_parent(sibling))
            end
        end
        if !isroot(node.parent)
            factor = combine(factor, message_from_parent(node.parent))
        end
        project(factor, domain(factor) ∩ node.domain)
    else
        node.message_from_parent
    end
end

# Compute the message
# μ i -> pa(i),
# caching intermediate computations.
function message_to_parent!(node::Architecture)
    @assert !isroot(node)
    if isnothing(node.message_to_parent)
        factor = node.factor
        for child in node.children
            factor = combine(factor, message_to_parent!(child))
        end
        node.message_to_parent = project(factor, domain(factor) ∩ node.parent.domain)
    end
    node.message_to_parent
end

# Compute the message
# μ pa(i) -> i,
# caching intermediate computations.
function message_from_parent!(node::Architecture)
    @assert !isroot(node)
    if isnothing(node.message_from_parent)
        factor = node.factor
        for sibling in node.parent.children
            if node.id != sibling.id
                factor = combine(factor, message_to_parent!(sibling))
            end
        end
        if !isroot(node.parent)
            factor = combine(factor, message_from_parent!(node.parent))
        end
        node.message_from_parent = project(factor, domain(factor) ∩ node.domain)
    end
    node.message_from_parent
end

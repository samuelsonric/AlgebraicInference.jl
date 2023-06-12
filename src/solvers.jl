"""
    InferenceSolver{T₁, T₂}

This is the type constructed by [`init(ip::InferenceProblem)`](@ref). Use it with
[`solve`](@ref) or [`solve!`](@ref) to solve inference problems.

An `InferenceSolver` can be reused to answer multiple queries:
```
is = init(ip)
sol1 = solve(is)
is.query = query2
sol2 = solve(is)
```
"""
mutable struct InferenceSolver{T₁, T₂}
    jt::JoinTree{T₁, T₂}
    query::Vector{T₂}
end

"""
    UWDSolver{T₁, T₂} = InferenceSolver{UWDBox{T₁, T₂}, T₂}

Solves a `UWDProblem`.
"""
const UWDSolver{T₁, T₂} = InferenceSolver{UWDBox{T₁, T₂}, T₂}

"""
    solve(is::InferenceSolver)

Solve an inference problem.
"""
function solve(is::InferenceSolver{T}) where T
    dom = collect(Set(is.query))
    for node in PreOrderDFS(is.jt)
        if dom ⊆ node.domain
            factor = node.factor
            for child in node.children
                factor = combine(factor, message_to_parent(child)::T)
            end
            if !isroot(node)
                factor = combine(factor, message_from_parent(node)::T)
            end
            return duplicate(project(factor, dom), is.query)
        end 
    end
    error("Query not covered by join tree.")
end

"""
    solve!(is::InferenceSolver)

Solve an inference problem, caching intermediate computations.
"""
function solve!(is::InferenceSolver{T}) where T
    dom = collect(Set(is.query))
    for node in PreOrderDFS(is.jt)
        if dom ⊆ node.domain
            factor = node.factor
            for child in node.children
                factor = combine(factor, message_to_parent!(child)::T)
            end
            if !isroot(node)
                factor = combine(factor, message_from_parent!(node)::T)
            end
            return duplicate(project(factor, dom), is.query)
        end 
    end
    error("Query not covered by join tree.")
end

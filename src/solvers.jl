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
    jt::JoinTree{T₁}
    objects::T₂
    query::Vector{Int}
end

"""
    init(ip::InferenceProblem, alg)

Construct a solver for an inference problem. The options for `alg` are
- [`MinDegree()`](@ref)
- [`MinFill()`](@ref)
"""
init(ip::InferenceProblem, alg)

function init(ip::InferenceProblem{T}, ::MinDegree) where T
    pg = copy(ip.pg)
    for i₁ in 2:length(ip.query)
        for i₂ in 1:i₁ - 1
            if ip.query[i₁] != ip.query[i₂]
                add_edge!(pg, ip.query[i₁], ip.query[i₂])
            end
        end
    end
    order = mindegree!(copy(pg))
    InferenceSolver(JoinTree(ip.kb, pg, order), ip.objects, ip.query) 
end

function init(ip::InferenceProblem{T}, ::MinFill) where T
    pg = copy(ip.pg)
    for i₁ in 2:length(ip.query)
        for i₂ in 1:i₁ - 1
            if ip.query[i₁] != ip.query[i₂]
                add_edge!(pg, ip.query[i₁], ip.query[i₂])
            end
        end
    end
    order = minfill!(copy(pg))
    InferenceSolver(JoinTree(ip.kb, pg, order), ip.objects, ip.query) 
end

"""
    solve(ip::InferenceProblem, alg)

Solve an inference problem. The options for `alg` are
- [`MinDegree()`](@ref)
- [`MinFill()`](@ref)
"""
solve(ip::InferenceProblem, alg)

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
                factor = combine(factor, message_to_parent(child)::Valuation{T})
            end
            if !isroot(node)
                factor = combine(factor, message_from_parent(node)::Valuation{T})
            end
            return pull_onto(extend(project(factor, domain(factor) ∩ dom), is.objects, dom), is.query)
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
                factor = combine(factor, message_to_parent!(child)::Valuation{T})
            end
            if !isroot(node)
                factor = combine(factor, message_from_parent!(node)::Valuation{T})
            end
            return pull_onto(extend(project(factor, domain(factor) ∩ dom), is.objects, dom), is.query)
        end 
    end
    error("Query not covered by join tree.")
end

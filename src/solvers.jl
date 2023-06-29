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
    tree::JoinTree{T₁}
    objects::Vector{T₂}
    query::Vector{Int}
end

"""
    init(ip::InferenceProblem, alg)

Construct a solver for an inference problem. The options for `alg` are
- [`MinDegree()`](@ref)
- [`MinFill()`](@ref)
"""
init(ip::InferenceProblem, alg)

function init(ip::InferenceProblem{T₁, T₂}, ::MinDegree) where {T₁, T₂}
    graph = copy(ip.graph)
    for i₁ in 2:length(ip.query)
        for i₂ in 1:i₁ - 1
            if ip.query[i₁] != ip.query[i₂]
                add_edge!(graph, ip.query[i₁], ip.query[i₂])
            end
        end
    end
    order = mindegree!(copy(graph))
    tree = JoinTree{T₁}(ip.factors, ip.objects, graph, order)
    InferenceSolver{T₁, T₂}(tree, ip.objects, ip.query)
end

function init(ip::InferenceProblem{T₁, T₂}, ::MinFill) where {T₁, T₂}
    graph = copy(ip.graph)
    for i₁ in 2:length(ip.query)
        for i₂ in 1:i₁ - 1
            if ip.query[i₁] != ip.query[i₂]
                add_edge!(graph, ip.query[i₁], ip.query[i₂])
            end
        end
    end
    order = minfill!(copy(graph))
    tree = JoinTree{T₁}(ip.factors, ip.objects, graph, order)
    InferenceSolver{T₁, T₂}(tree, ip.objects, ip.query)
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
    variables = unique(is.query)
    for node in PreOrderDFS(is.tree)
        if variables ⊆ node.domain
            factor = node.factor
            for child in node.children
                message = message_to_parent(child, is.objects)::Valuation{T}
                factor = combine(factor, message, is.objects)
            end
            if !isroot(node)
                message = message_from_parent(node, is.objects)::Valuation{T}
                factor = combine(factor, message, is.objects)
            end
            factor = project(factor, domain(factor) ∩ variables, is.objects)
            factor = extend(factor, variables, is.objects)
            return expand(factor, is.query, is.objects)
        end 
    end
    error("Query not covered by join tree.")
end

"""
    solve!(is::InferenceSolver)

Solve an inference problem, caching intermediate computations.
"""
function solve!(is::InferenceSolver{T}) where T
    variables = unique(is.query)
    for node in PreOrderDFS(is.tree)
        if variables ⊆ node.domain
            factor = node.factor
            for child in node.children
                message = message_to_parent!(child, is.objects)::Valuation{T}
                factor = combine(factor, message, is.objects)
            end
            if !isroot(node)
                message = message_from_parent!(node, is.objects)::Valuation{T}
                factor = combine(factor, message, is.objects)
            end
            factor = project(factor, domain(factor) ∩ variables, is.objects)
            factor = extend(factor, variables, is.objects)
            return expand(factor, is.query, is.objects)
        end 
    end
    error("Query not covered by join tree.")
end

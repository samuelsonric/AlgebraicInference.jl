"""
    InferenceSolver{T₁, T₂}

A solver for an inference problem. 

An `InferenceSolver` can be reused to answer multiple queries:
```julia
is = init(ip)
sol₁ = solve(is)
is.query = query
sol₂ = solve(is)
```
"""
mutable struct InferenceSolver{T₁, T₂}
    tree::JoinTree{T₁}
    objects::Vector{T₂}
    query::Vector{Int}
end

"""
    elimination_order(graph::LabeledGraph, alg::EliminationAlgorithm)
"""
elimination_order(graph::LabeledGraph, alg::EliminationAlgorithm)

function elimination_order(graph::LabeledGraph, alg::MinDegree)
    mindegree!(copy(graph))
end

function elimination_order(graph::LabeledGraph, alg::MinFill)
    minfill!(copy(graph))
end

"""
    init(ip::InferenceProblem, alg::EliminationAlgorithm)

Construct a solver for an inference problem.
"""
CommonSolve.init(ip::InferenceProblem, alg::EliminationAlgorithm)

function CommonSolve.init(ip::InferenceProblem{T₁, T₂}, alg::EliminationAlgorithm) where {T₁, T₂}
    model = copy(ip.model)
    context!(model, ip.evidence)

    for i₁ in eachindex(ip.query), i₂ in 1:i₁ - 1
        if ip.query[i₁] != ip.query[i₂]
            Graphs.add_edge!(model.graph, ip.query[i₁], ip.query[i₂])
        end
    end

    order = elimination_order(model.graph, alg) 
    tree = JoinTree{T₁}(model.factors, model.graph, order)

    InferenceSolver{T₁, T₂}(tree, model.objects, ip.query)
end

"""
    solve(is::InferenceSolver)

Solve an inference problem.
"""
function CommonSolve.solve(is::InferenceSolver{T}) where T
    for node in PreOrderDFS(is.tree)
        if is.query ⊆ node.variables
            factor = reduce(node.factors; init=zero(Factor{T})) do fac₁, fac₂
                combine(fac₁, fac₂, is.objects)
            end

            for child in node.children
                message = message_to_parent(child, is.objects)::Factor{T}
                factor = combine(factor, message, is.objects)
            end

            if !isroot(node)
                message = message_from_parent(node, is.objects)::Factor{T}
                factor = combine(factor, message, is.objects)
            end

            factor = project(factor, is.query, is.objects)

            return permute(factor, is.query, is.objects)
        end 
    end

    error("Query not covered by join tree.")
end

"""
    solve!(is::InferenceSolver)

Solve an inference problem, caching intermediate computations.
"""
function CommonSolve.solve!(is::InferenceSolver{T}) where T
    for node in PreOrderDFS(is.tree)
        if is.query ⊆ node.variables
            factor = reduce(node.factors; init=zero(Factor{T})) do fac₁, fac₂
                combine(fac₁, fac₂, is.objects)
            end

            for child in node.children
                message = message_to_parent!(child, is.objects)::Factor{T}
                factor = combine(factor, message, is.objects)
            end

            if !isroot(node)
                message = message_from_parent!(node, is.objects)::Factor{T}
                factor = combine(factor, message, is.objects)
            end

            factor = project(factor, is.query, is.objects)
    
            return permute(factor, is.query, is.objects)
        end 
    end

    error("Query not covered by join tree.")
end

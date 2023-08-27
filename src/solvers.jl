"""
    InferenceSolver{T₁, T₂, T₃}

A solver for an inference problem. 

An `InferenceSolver` can be reused to answer multiple queries:
```julia
is = init(ip)
sol₁ = solve(is)
is.query = query
sol₂ = solve(is)
```
"""
mutable struct InferenceSolver{T₁, T₂, T₃}
    model::JoinTreeModel{T₁, T₂, T₃}
    query::Vector{T₁}
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

function CommonSolve.init(ip::InferenceProblem, alg::EliminationAlgorithm)
    model = copy(ip.model)
    observe!(model, ip.evidence)

    for i₁ in eachindex(ip.query), i₂ in 1:i₁ - 1
        if ip.query[i₁] != ip.query[i₂]
            Graphs.add_edge!(model.graph, ip.query[i₁], ip.query[i₂])
        end
    end

    order = elimination_order(model.graph, alg)

    InferenceSolver(JoinTreeModel(model, order), ip.query)
end

"""
    solve(is::InferenceSolver)

Solve an inference problem.
"""
function CommonSolve.solve(is::InferenceSolver{T₁, T₂}) where {T₁, T₂}
    for node in PreOrderDFS(is.model.tree)
        if is.query ⊆ node.variables
            factor = reduce(is.model.factors[node.factors]; init=zero(Factor{T₁, T₂})) do fac₁, fac₂
                combine(fac₁, fac₂, is.model.objects)
            end

            for child in node.children
                message = message_to_parent(is.model, child)::Factor{T₁, T₂}
                factor = combine(factor, message, is.model.objects)
            end

            if !isroot(node)
                message = message_from_parent(is.model, node)::Factor{T₁, T₂}
                factor = combine(factor, message, is.model.objects)
            end

            factor = project(factor, is.query, is.model.objects)

            return permute(factor, is.query, is.model.objects)
        end 
    end

    error("Query not covered by join tree.")
end

"""
    solve!(is::InferenceSolver)

Solve an inference problem, caching intermediate computations.
"""
function CommonSolve.solve!(is::InferenceSolver{T₁, T₂}) where {T₁, T₂}
    for node in PreOrderDFS(is.model.tree)
        if is.query ⊆ node.variables
            factor = reduce(is.model.factors[node.factors]; init=zero(Factor{T₁, T₂})) do fac₁, fac₂
                combine(fac₁, fac₂, is.model.objects)
            end

            for child in node.children
                message = message_to_parent!(is.model, child)::Factor{T₁, T₂}
                factor = combine(factor, message, is.model.objects)
            end

            if !isroot(node)
                message = message_from_parent!(is.model, node)::Factor{T₁, T₂}
                factor = combine(factor, message, is.model.objects)
            end

            factor = project(factor, is.query, is.model.objects)
    
            return permute(factor, is.query, is.model.objects)
        end 
    end

    error("Query not covered by join tree.")
end

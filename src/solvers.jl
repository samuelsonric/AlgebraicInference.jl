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
    architecture::SSArchitecture{T₁, T₂, T₃}
    query::Vector{T₁}
end


"""
    init(ip::InferenceProblem, alg::EliminationAlgorithm)

Construct a solver for an inference problem.
"""
function CommonSolve.init(ip::InferenceProblem, alg::EliminationAlgorithm)
    model = copy(ip.model)
    observe!(model, ip.evidence)

    for i₁ in eachindex(ip.query), i₂ in 1:i₁ - 1
        v₁ = model.labels.index[ip.query[i₁]]
        v₂ = modle.labels.index[ip.query[i₂]]

        if v₁ != v₂
            Graphs.add_edge!(model.graph, v₁, v₂)
        end
    end

    order = EliminationOrder(model.graph, alg)
    architecture = SSArchitecture(model, order)

    InferenceSolver(architecture, ip.query)
end


"""
    solve!(is::InferenceSolver)

Solve an inference problem, caching intermediate computations.
"""
function CommonSolve.solve!(is::InferenceSolver)
    solve!(is.architecture, is.query)
end

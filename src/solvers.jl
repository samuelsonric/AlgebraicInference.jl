"""
    InferenceSolver{T₁, T₂, T₃, T₄, T₅}

A solver for an inference problem. 

An `InferenceSolver` can be reused to answer multiple queries:
```julia
is = init(ip)
sol₁ = solve(is)
is.query = query
sol₂ = solve(is)
```
"""
mutable struct InferenceSolver{T₁, T₂, T₃, T₄, T₅}
    architecture::Architecture{T₁, T₂, T₃, T₄}
    architecture_type::T₅
    query::Vector{T₁}
end


# Construct a solver for an inference problem.
function InferenceSolver(
    problem::InferenceProblem,
    elimination_algorithm::EliminationAlgorithm,
    supernode_type::SupernodeType,
    architecture_type::ArchitectureType)

    model = reduce_to_context(problem.model, problem.context)

    for i₁ in eachindex(problem.query), i₂ in 1:i₁ - 1
        v₁ = model.labels.index[problem.query[i₁]]
        v₂ = model.labels.index[problem.query[i₂]]

        if v₁ != v₂
            Graphs.add_edge!(model.graph, v₁, v₂)
        end
    end

    architecture = Architecture(model, elimination_algorithm, supernode_type)
    InferenceSolver(architecture, architecture_type, problem.query)
end


"""
    init(
        problem::InferenceProblem,
        elimination_algorithm::EliminationAlgorithm=MinFill(),
        supernode_type::SupernodeType=Node(),
        architecture_type::ArchitectureType=ShenoyShafer())

Construct a solver for an inference problem.
"""
function CommonSolve.init(
    problem::InferenceProblem,
    elimination_algorithm::EliminationAlgorithm=MinFill(),
    supernode_type::SupernodeType=Node(),
    architecture_type::ArchitectureType=ShenoyShafer())

    InferenceSolver(problem, elimination_algorithm, supernode_type, architecture_type)
end


"""
    solve!(solver::InferenceSolver)

Solve an inference problem, caching intermediate computations.
"""
function CommonSolve.solve!(solver::InferenceSolver)
    solve!(solver.architecture, solver.architecture_type, solver.query)
end


"""
    mean(solver::InferenceSolver)
"""
function Statistics.mean(solver::InferenceSolver)
    mean(solver.architecture)
end


"""
    rand(rng::AbstractRNG=default_rng(), solver::InferenceSolver)
"""
function Base.rand(rng::AbstractRNG, solver::InferenceSolver)
    rand(rng, solver.architecture)
end


function Base.rand(solver::InferenceSolver)
    rand(default_rng(), solver)
end

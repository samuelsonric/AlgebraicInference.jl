"""
    InferenceSolver{T₁, T₂, T₃, T₄, T₅}

A solver for an inference problem. A solver can use one of four message-passing algorithms:
- The Shenoy-Shafer architecture
- The Lauritzen-Spiegelhalter architecture
- The HUGIN architecture
- The idempotent architecture
"""
mutable struct InferenceSolver{T₁, T₂, T₃, T₄, T₅}
    architecture::Architecture{T₁, T₂, T₃, T₄, T₅}
    query::Vector{T₂}
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

    architecture = Architecture(
        model,
        elimination_algorithm,
        supernode_type,
        architecture_type)

    InferenceSolver(architecture, problem.query)
end


function InferenceSolver(
    problem::InferenceProblem,
    elimination_algorithm::EliminationAlgorithm,
    supernode_type::SupernodeType,
    architecture_type::AncestralSampler)

    model = reduce_to_context(problem.model, problem.context)

    architecture = Architecture(
        model,
        elimination_algorithm,
        supernode_type,
        architecture_type)

    InferenceSolver(architecture, problem.query)
end


"""
    init(
        problem::InferenceProblem,
        elimination_algorithm::EliminationAlgorithm=MinFill(),
        supernode_type::SupernodeType=Node(),
        architecture_type::ArchitectureType=ShenoyShafer())

Construct a solver for an inference problem. This involves several steps:
```
    interaction graph
        ↓ elimination algorithm
    elimination tree
        ↓ supernode type
    join tree
```

The argument `architecture_type` deterimines which message-passing algorithm is used to
solve the inference problem.
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

Solve an inference problem.
"""
function CommonSolve.solve!(solver::InferenceSolver)
    solve!(solver.architecture, solver.query)
end


function CommonSolve.solve!(solver::InferenceSolver{AncestralSampler()})
    solve!(solver.architecture)
    solver
end


"""
    mean(solver::InferenceSolver{AncestralSampler()})
"""
function Statistics.mean(solver::InferenceSolver{AncestralSampler()})
    mean(solver.architecture, solver.query)
end


"""
    rand(rng::AbstractRNG, solver::InferenceSolver{AncestralSampler()})
"""
function Base.rand(rng::AbstractRNG, solver::InferenceSolver{AncestralSampler()})
    rand(rng, solver.architecture, solver.query)
end


"""
    rand(solver::InferenceSolver{AncestralSampler()})
"""
function Base.rand(solver::InferenceSolver{AncestralSampler()})
    rand(default_rng(), solver)
end

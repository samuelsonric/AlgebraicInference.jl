"""
    InferenceSolver{T₁, T₂, T₃, T₄, T₅}

A solver for an inference problem.
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

First an elimination agorithm is used to construct an elimination tree from the model's
interaction graph. Then, nodes of the elimination tree are merged into supernodes, forming
a join (or junction) tree.

When [`solve!`](@ref) is called on the solver, a message passing algorithm is used to compute
the solution.
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

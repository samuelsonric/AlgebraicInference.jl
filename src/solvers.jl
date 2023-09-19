"""
    InferenceSolver{T₁, T₂, T₃, T₄}

A solver for an inference problem. 

An `InferenceSolver` can be reused to answer multiple queries:
```julia
is = init(ip)
sol₁ = solve(is)
is.query = query
sol₂ = solve(is)
```
"""
mutable struct InferenceSolver{T₁, T₂, T₃, T₄}
    architecture::Architecture{T₁, T₂, T₃, T₄}
    query::Vector{T₁}
end


# Construct a solver for an inference problem.
function InferenceSolver(
    problem::InferenceProblem,
    elalg::EliminationAlgorithm,
    stype::SupernodeType,
    atype::ArchitectureType)

    model = copy(problem.model)
    observe!(model, problem.context)

    for i₁ in eachindex(problem.query), i₂ in 1:i₁ - 1
        v₁ = model.labels.index[problem.query[i₁]]
        v₂ = model.labels.index[problem.query[i₂]]

        if v₁ != v₂
            Graphs.add_edge!(model.graph, v₁, v₂)
        end
    end

    architecture = Architecture(model, elalg, stype, atype)
    InferenceSolver(architecture, problem.query)
end


"""
    init(
        problem::InferenceProblem,
        elalg::EliminationAlgorithm=MinFill(),
        stype::SupernodeType=Node(),
        atype::ArchitectureType=ShenoyShafer())

Construct a solver for an inference problem.
"""
function CommonSolve.init(
    problem::InferenceProblem,
    elalg::EliminationAlgorithm=MinFill(),
    stype::SupernodeType=Node(),
    atype::ArchitectureType=ShenoyShafer())

    InferenceSolver(problem, elalg, stype, atype)
end


"""
    solve!(solver::InferenceSolver)

Solve an inference problem, caching intermediate computations.
"""
function CommonSolve.solve!(solver::InferenceSolver)
    solve!(solver.architecture, solver.query)
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

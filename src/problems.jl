"""
    InferenceProblem{T₁, T₂, T₃, T₄}

An inference problem computes the conditional distribution of ``X`` given ``Y = e``,
where ``X`` and ``Y`` are random variables whose joint probability distribution is
specified by a graphical model.
"""
mutable struct InferenceProblem{T₁, T₂, T₃, T₄}
    model::GraphicalModel{T₁, T₂, T₃}
    evidence::Dict{T₁, T₄}
    query::Vector{T₁}
end

"""
    InferenceProblem{T₁, T₂, T₃, T₄}(
        wd::AbstractUWD,
        hom_map::AbstractDict,
        ob_map::AbstractDict;
        hom_attr::Symbol=:name,
        ob_attr::Symbol=:variable) where {T₁, T₂, T₃, T₄}

Construct an inference problem that performs undirected compositon.
"""
function InferenceProblem{T₁, T₂, T₃, T₄}(
    wd::AbstractUWD,
    hom_map::AbstractDict,
    ob_map::AbstractDict;
    hom_attr::Symbol=:name,
    ob_attr::Symbol=:variable) where {T₁, T₂, T₃, T₄}

    homs = [hom_map[x] for x in subpart(wd, hom_attr)]
    obs = [ob_map[x] for x in subpart(wd, ob_attr)]
    InferenceProblem{T₁, T₂, T₃, T₄}(wd, homs, obs)
end

"""
    InferenceProblem{T₁, T₂, T₃, T₄}(
        wd::AbstractUWD,
        homs::AbstractVector,
        obs::AbstractVector) where {T₁, T₂, T₃, T₄}

Construct an inference problem that performs undirected compositon.
"""
function InferenceProblem{T₁, T₂, T₃, T₄}(
    wd::AbstractUWD,
    homs::AbstractVector,
    obs::AbstractVector) where {T₁, T₂, T₃, T₄}

    fg = @migrate UndirectedBipartiteGraph wd begin
        E  => Port
        V₁ => Box
        V₂ => Junction

        src => box
        tgt => junction
    end

    model = GraphicalModel{T₁, T₂, T₃}(fg, homs, obs)
    evidence = Dict{T₁, T₄}()
    query = Vector{T₁}(subpart(wd, :outer_junction))

    InferenceProblem(model, evidence, query)
end

"""
    InferenceProblem{T₁, T₂, T₃, T₄}(
        bn::BayesNet,
        query::AbstractVector,
        evidence::AbstractDict) where {T₁, T₂, T₃, T₄}
"""
function InferenceProblem{T₁, T₂, T₃, T₄}(
    bn::BayesNet,
    query::AbstractVector,
    evidence::AbstractDict) where {T₁, T₂, T₃, T₄}

    model = GraphicalModel{T₁, T₂, T₃}(bn)
    evidence = Dict{T₁, T₄}(bn.name_to_index[l] => [v] for (l, v) in evidence) 
    query = T₁[bn.name_to_index[l] for l in query]
   
    InferenceProblem(model, evidence, query) 
end

"""
    solve(ip::InferenceProblem, alg::EliminationAlgorithm)

Solve an inference problem.
"""
CommonSolve.solve(ip::InferenceProblem, alg::EliminationAlgorithm)

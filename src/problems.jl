"""
    InferenceProblem{T₁, T₂, T₃}

An inference problem computes the conditional distribution
```math
    X \\mid (Y = e),
```
where the joint distribution of ``X`` and ``Y`` is specified by an undirected graphical model.
"""
mutable struct InferenceProblem{T₁, T₂, T₃}
    model::GraphicalModel{T₁, T₂}
    evidence::Dict{Int, T₃}
    query::Vector{Int}
end

"""
    InferenceProblem{T₁, T₂}(wd::AbstractUWD, hom_map::AbstractDict, ob_map::AbstractDict;
        hom_attr=:name, ob_attr=:variable) where {T₁, T₂}

Construct an inference problem that performs undirected compositon.
"""
function InferenceProblem{T₁, T₂, T₃}(wd::AbstractUWD, hom_map::AbstractDict, ob_map::AbstractDict;
    hom_attr::Symbol=:name, ob_attr::Symbol=:variable) where {T₁, T₂, T₃}
    homs = [hom_map[x] for x in subpart(wd, hom_attr)]
    obs = [ob_map[x] for x in subpart(wd, ob_attr)]
    InferenceProblem{T₁, T₂, T₃}(wd, homs, obs)
end

"""
    InferenceProblem{T₁, T₂}(wd::AbstractUWD, homs::AbstractVector,
        obs::AbstractVector) where {T₁, T₂}

Construct an inference problem that performs undirected compositon.
"""
function InferenceProblem{T₁, T₂, T₃}(wd::AbstractUWD, homs::AbstractVector,
    obs::AbstractVector) where {T₁, T₂, T₃}

    @assert nboxes(wd) == length(homs)
    @assert njunctions(wd) == length(obs)

    boxes = collect(subpart(wd, :box))
    juncs = collect(subpart(wd, :junction))
    query = collect(subpart(wd, :outer_junction))

    model = GraphicalModel{T₁, T₂}(nboxes(wd), njunctions(wd))

    i = 1

    for i₁ in 2:length(juncs)
        for i₂ in i:i₁ - 1
            if juncs[i₁] != juncs[i₂]
                Graphs.add_edge!(model.graph, juncs[i₁], juncs[i₂])
            end
        end

        if boxes[i] != boxes[i₁]
            model.factors[boxes[i]] = Factor(homs[boxes[i]], juncs[i:i₁ - 1])
            i = i₁
        end
    end

    model.factors[end] = Factor(homs[end], juncs[i:end])
    model.objects .= obs

    InferenceProblem(model, Dict{Int, T₃}(), query)
end

"""
    InferenceProblem{T₁, T₂}(bn::BayesNet, query::AbstractVector,
        evidence::AbstractDict) where {T₁, T₂}
"""
function InferenceProblem{T₁, T₂, T₃}(bn::BayesNet, query::AbstractVector,
    evidence::AbstractDict) where {T₁, T₂, T₃}

    n = length(bn)
    model = GraphicalModel{T₁, T₂}(n, n)
    model.objects .= 1

    for i in 1:n
        cpd = bn.cpds[i]; l = name(cpd)
        pas = [bn.name_to_index[l] for l in parents(cpd)] 
 
        model.factors[i] = Factor(cpd, [pas; i])
 
        for j₁ in eachindex(pas)
            Graphs.add_edge!(model.graph, pas[j₁], i)

            for j₂ in 1:j₁ - 1
                Graphs.add_edge!(model.graph, pas[j₁], pas[j₂])
            end
        end
    end

    evidence = Dict(bn.name_to_index[l] => [v] for (l, v) in evidence) 
    query = [bn.name_to_index[l] for l in query]

    InferenceProblem{T₁, T₂, T₃}(model, evidence, query) 
end

"""
    solve(ip::InferenceProblem, alg::EliminationAlgorithm)

Solve an inference problem.
"""
CommonSolve.solve(ip::InferenceProblem, alg::EliminationAlgorithm)

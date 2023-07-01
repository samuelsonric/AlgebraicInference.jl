"""
    InferenceProblem{T₁, T₂}

An inference problem over a valuation algebra. Construct a solver for an inference problem
with the function [`init`](@ref), or solve it directly with [`solve`](@ref).
"""
mutable struct InferenceProblem{T₁, T₂}
    factors::Vector{Valuation{T₁}}
    objects::Vector{T₂}
    graph::Graph{Int}
    query::Vector{Int}
end

"""
    MinDegree

Contructs a covering join tree for an inference problem using the variable elimination
algorithm. Variables are eliminated according to the "minimum degree" heuristic.
"""
struct MinDegree end

"""
    MinFill

Contructs a covering join tree for an inference problem using the variable elimination
algorithm. Variables are eliminated according to the "minimum fill" heuristic.
"""
struct MinFill end

"""
    InferenceProblem{T₁, T₂}(wd::AbstractUWD, hom_map::AbstractDict, ob_map::AbstractDict;
        hom_attr=:name, ob_attr=:variable) where {T₁, T₂}

Construct an inference problem that performs undirected composition. Before being composed,
the values of `hom_map` are converted to type `T₁`, and the values of `ob_map` are converted
to type `T₂`.
"""
function InferenceProblem{T₁, T₂}(wd::AbstractUWD, hom_map::AbstractDict, ob_map::AbstractDict;
    hom_attr=:name, ob_attr=:variable) where {T₁, T₂}
    homs = [hom_map[x] for x in subpart(wd, hom_attr)]
    obs = [ob_map[x] for x in subpart(wd, ob_attr)]
    InferenceProblem{T₁, T₂}(wd, homs, obs)
end

"""
    InferenceProblem{T₁, T₂}(wd::AbstractUWD, homs::AbstractVector,
        obs::AbstractVector) where {T₁, T₂}

Construct an inference problem that performs undirected composition. Before being composed,
the elements of `homs` are converted to type `T₁`, and the elements of `obs` are converted
to type `T₂`.
"""
function InferenceProblem{T₁, T₂}(wd::AbstractUWD, homs::AbstractVector,
    obs::AbstractVector) where {T₁, T₂}
    @assert nboxes(wd) == length(homs)
    @assert njunctions(wd) == length(obs)
    boxes = collect(subpart(wd, :box))
    juncs = collect(subpart(wd, :junction))
    query = collect(subpart(wd, :outer_junction))
    factors = Vector{Valuation{T₁}}(undef, nboxes(wd))
    graph = Graph(njunctions(wd))
    i = 1
    for i₁ in 2:length(juncs)
        for i₂ in i:i₁ - 1
            if juncs[i₁] != juncs[i₂]
                add_edge!(graph, juncs[i₁], juncs[i₂])
            end
        end
        if boxes[i] != boxes[i₁]
            factors[boxes[i]] = contract(T₁, homs[boxes[i]], juncs[i:i₁ - 1], obs)
            i = i₁
        end
    end
    factors[end] = contract(T₁, homs[end], juncs[i:end], obs)
    InferenceProblem{T₁, T₂}(factors, obs, graph, query)
end

"""
    InferenceProblem{T₁, T₂}(bn::BayesNet, query::AbstractVector,
        evidence::AbstractDict) where {T₁, T₂}
"""
function InferenceProblem{T₁, T₂}(bn::BayesNet, query::AbstractVector,
    evidence::AbstractDict) where {T₁, T₂}
    n = length(bn)
    factors = Vector{Valuation{T₁}}(undef, n)
    objects = ones(T₂, n)
    graph = Graph(bn.dag)
    for i in 1:n
        cpd = bn.cpds[i]; l = name(cpd)
        pa = map(l -> bn.name_to_index[l], parents(cpd))
        for j₁ in 2:length(pa)
            for j₂ in 1:j₁ - 1
                add_edge!(graph, pa[j₁], pa[j₂])
            end
        end
        factor = Valuation{T₁}(cpd, [pa; i])
        if haskey(evidence, l)
            observation = Valuation{T₁}(evidence[l], [i])
            factor = combine(factor, observation, objects)
        end
        factors[i] = factor
    end
    query = map(l -> bn.name_to_index[l], query)
    InferenceProblem{T₁, T₂}(factors, objects, graph, query)
end

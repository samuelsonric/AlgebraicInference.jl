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
    query = collect(subpart(wd, :outer_junction))
    ports = collect(subpart(wd, :junction))
    factors = Vector{Valuation{T₁}}(undef, nboxes(wd))
    graph = Graph(njunctions(wd))
    i = 1
    for i₁ in 2:length(ports)
        for i₂ in i:i₁ - 1
            if ports[i₁] != ports[i₂]
                add_edge!(graph, ports[i₁], ports[i₂])
            end
        end
        if box(wd, i) != box(wd, i₁)
            factors[box(wd, i)] = Valuation{T₁}(homs[box(wd, i)], ports[i:i₁ - 1])
            i = i₁
        end
    end
    factors[end] = Valuation{T₁}(homs[end], ports[i:end])
    InferenceProblem{T₁, T₂}(factors, obs, graph, query)
end

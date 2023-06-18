"""
    InferenceProblem{T₁, T₂}

An inference problem over a valuation algebra. Construct a solver for an inference problem
with the function [`init`](@ref), or solve it directly with [`solve`](@ref).
"""
mutable struct InferenceProblem{T₁, T₂}
    kb::Vector{Valuation{T₁}}
    objects::T₂
    pg::Graph{Int}
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
    InferenceProblem{T}(wd::AbstractUWD, hom_map::AbstractDict,
        ob_map::Union{Nothing, AbstractDict}=nothing;
        hom_attr=:name, ob_attr=:variable) where T

Construct an inference problem that performs undirected composition. Before being composed,
the values of `hom_map` are converted to type `T`.
"""
function InferenceProblem{T}(wd::AbstractUWD, hom_map::AbstractDict,
    ob_map::Union{Nothing, AbstractDict}=nothing;
    hom_attr=:name, ob_attr=:variable) where T
    homs = [hom_map[x] for x in subpart(wd, hom_attr)]
    obs = isnothing(ob_map) ? nothing : [ob_map[x] for x in subpart(wd, ob_attr)]
    InferenceProblem{T}(wd, homs, obs)
end

"""
    InferenceProblem{T}(wd::AbstractUWD, homs::AbstractVector,
        obs::Union{Nothing, AbstractVector}=nothing) where T

Construct an inference problem that performs undirected composition. Before being composed,
the elements of `homs` are converted to type `T`.
"""
function InferenceProblem{T}(wd::AbstractUWD, homs::AbstractVector,
    obs::Union{Nothing, AbstractVector}=nothing) where T
    @assert nboxes(wd) == length(homs)
    @assert isnothing(obs) || njunctions(wd) == length(obs)
    query = collect(subpart(wd, :outer_junction))
    ports = collect(subpart(wd, :junction))
    kb = Vector{Valuation{T}}(undef, nboxes(wd))
    pg = Graph(njunctions(wd))
    i = 1
    for i₁ in 2:length(ports)
        for i₂ in i:i₁ - 1
            if ports[i₁] != ports[i₂]
                add_edge!(pg, ports[i₁], ports[i₂])
            end
        end
        if box(wd, i) != box(wd, i₁)
            kb[box(wd, i)] = Valuation{T}(homs[box(wd, i)], ports[i:i₁ - 1], false)
            i = i₁
        end
    end
    kb[end] = Valuation{T}(homs[end], ports[i:end], false)
    InferenceProblem(kb, obs, pg, query)
end

"""
    InferenceProblem{T}

An inference problem over a valuation algebra. Construct a solver for an inference problem
with the function [`init`](@ref), or solve it directly with [`solve`](@ref).
"""
mutable struct InferenceProblem{T}
    kb::Vector{Valuation{T}}
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
    InferenceProblem{T}(wd::AbstractUWD, hom_map::AbstractDict, ob_map::AbstractDict;
        hom_attr=:name, ob_attr=:variable) where T

Construct an inference problem that performs undirected composition. Before being composed,
the values of `hom_map` are converted to type `T`.
"""
function InferenceProblem{T}(wd::AbstractUWD, hom_map::AbstractDict, ob_map::AbstractDict;
    hom_attr=:name, ob_attr=:variable) where T
    homs = [hom_map[x] for x in subpart(wd, hom_attr)]
    obs = [ob_map[x] for x in subpart(wd, ob_attr)]
    InferenceProblem{T}(wd, homs, obs)
end

# For PROPs
function InferenceProblem{T}(wd::AbstractUWD, hom_map::AbstractDict; hom_attr=:name) where T
    homs = [hom_map[x] for x in subpart(wd, hom_attr)]
    InferenceProblem{T}(wd, homs)
end

"""
    InferenceProblem{T}(wd::AbstractUWD, homs, obs) where T

Construct an inference problem that performs undirected composition. Before being composed,
the elements of `homs` are converted to type `T`.
"""
function InferenceProblem{T}(wd::AbstractUWD, homs, obs) where T
    @assert nboxes(wd) == length(homs)
    @assert njunctions(wd) == length(obs)
    query = collect(subpart(wd, :outer_junction))
    js = collect(subpart(wd, :junction))
    kb = Vector{Valuation{T}}(undef, nboxes(wd))
    pg = Graph(njunctions(wd))
    pt = 1
    for i in ports(wd)::UnitRange{Int}
        for j in pt:i - 1
            if js[i] != js[j]
                add_edge!(pg, js[i], js[j])
            end
        end
        if box(wd, pt) != box(wd, i)
            kb[box(wd, pt)] = Valuation{T}(homs[box(wd, pt)], js[pt:i - 1], false)
            pt = i
        end
    end
    kb[end] = Valuation{T}(homs[end], js[pt:end], false)
    for label in setdiff(query, js)
        push!(kb, one(Valuation{T}, obs[label], label))
    end
    InferenceProblem{T}(kb, pg, query)
end

# For PROPs
function InferenceProblem{T}(wd::AbstractUWD, homs::AbstractVector) where T
    InferenceProblem{T}(wd, homs, ones(Int, njunctions(wd)))
end

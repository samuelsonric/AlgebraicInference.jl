"""
    InferenceProblem{T₁, T₂ <: Valuation{T₁}}
"""
mutable struct InferenceProblem{T₁, T₂ <: Valuation{T₁}}
    query::Vector{T₁}
    kb::Vector{T₂}

    @doc """
        InferenceProblem{T₁, T₂}(query, kb)
    """
    function InferenceProblem{T₁, T₂}(query, kb) where {T₁, T₂ <: Valuation{T₁}}
        new{T₁, T₂}(query, kb)
    end
end

"""
    UWDProblem{T₁, T₂} = InferenceProblem{T₁, UWDBox{T₁, T₂}}
"""
const UWDProblem{T₁, T₂} = InferenceProblem{T₁, UWDBox{T₁, T₂}}

"""
    MinWidth
"""
struct MinWidth end

"""
    MinFill
"""
struct MinFill end

"""
    InferenceProblem(query, kb)
"""
function InferenceProblem(query, kb)
    T₁ = eltype(query)
    T₂ = eltype(kb)
    InferenceProblem{T₁, T₂}(query, kb)
end

"""
    UWDProblem(wd::AbstractUWD, bm::AbstractDict)
"""
function UWDProblem(wd::AbstractUWD, bm::AbstractDict{<:Any, T}) where T
    UWDProblem{Int, T}(wd, bm)
end

"""
    UWDProblem{T₁, T₂}(wd::AbstractUWD, bm::AbstractDict) where {T₁, T₂}
"""
function UWDProblem{T₁, T₂}(wd::AbstractUWD, bm::AbstractDict) where {T₁, T₂}
    bs = T₂[bm[x] for x in subpart(wd, :name)]
    UWDProblem{T₁, T₂}(wd, bs)
end

"""
    UWDProblem(wd::AbstractUWD, bs)

Translate an undirected wiring diagram
```math
    B \\xleftarrow{\\mathtt{box}} P \\xrightarrow{\\mathtt{junc}} J
    \\xleftarrow{\\mathtt{junc'}} Q
```
into an inference problem in a valuation algebra.

The diagram must satisfy the following constraints:
- ``\\mathtt{junc'}`` is injective.
- ``\\mathtt{image}(\\mathtt{junc'}) \\subseteq \\mathtt{image}(\\mathtt{junc})``
- For all ``x, y \\in P``, ``\\mathtt{box}(x) = \\mathtt{box}(y)`` and
  ``\\mathtt{junc}(x) = \\mathtt{junc}(y)`` implies that ``x = y``. 
"""
function UWDProblem(wd::AbstractUWD, bs)
    T = eltype(bs)
    UWDProblem{Int, T}(wd, bs)
end

"""
    UWDProblem{T₁, T₂}(wd::AbstractUWD, boxes) where {T₁, T₂}
"""
function UWDProblem{T₁, T₂}(wd::AbstractUWD, bs) where {T₁, T₂}
    UWDProblem{T₁, T₂}(wd, collect(bs))
end

function UWDProblem(
    wd::UntypedRelationDiagram{<:Any, T₁},
    bm::AbstractDict{<:Any, T₂}) where {T₁, T₂}

    UWDProblem{T₁, T₂}(wd, bm)
end

function UWDProblem{T₁, T₂}(wd::AbstractUWD, bs::AbstractVector) where {T₁, T₂}
    @assert nboxes(wd) == length(bs)
    ls = [T₁[] for box in bs]
    for i in ports(wd; outer=false)
        push!(ls[box(wd, i)], junction(wd, i; outer=false))
    end
    query = [
        junction(wd, i; outer=true)
        for i in ports(wd; outer=true)]
    kb = [
        UWDBox{T₁, T₂}(labels, box)
        for (labels, box) in zip(ls, bs)]
    UWDProblem{T₁, T₂}(query, kb)
end

function UWDProblem{T₁, T₂}(wd::UntypedRelationDiagram, bs::AbstractVector) where {T₁, T₂}
    @assert nboxes(wd) == length(bs)
    ls = [T₁[] for box in bs]
    for i in ports(wd; outer=false)
        push!(ls[box(wd, i)], subpart(wd, junction(wd, i; outer=false), :variable))
    end
    query = [
        subpart(wd, junction(wd, i; outer=true), :variable)
        for i in ports(wd; outer=true)]
    kb = [
        UWDBox{T₁, T₂}(labels, box)
        for (labels, box) in zip(ls, bs)]
    UWDProblem{T₁, T₂}(query, kb)
end

"""
    init(ip::InferenceProblem, alg)
"""
init(ip::InferenceProblem, alg)

function init(ip::InferenceProblem{T₁, T₂}, ::MinWidth) where {T₁, T₂}
    JoinTree{T₁, T₂}(ip.kb, minwidth!(primal_graph(ip.kb), ip.query))
end

function init(ip::InferenceProblem{T₁, T₂}, ::MinFill) where {T₁, T₂}
    JoinTree{T₁, T₂}(ip.kb, minfill!(primal_graph(ip.kb), ip.query))
end

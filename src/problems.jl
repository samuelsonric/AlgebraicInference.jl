"""
    InferenceProblem{T₁, T₂ <: Valuation{T₁}}
"""
mutable struct InferenceProblem{T₁, T₂ <: Valuation{T₁}}
    query::Vector{T₁}
    kb::Vector{T₂}
end

"""
    MinWidth
"""
struct MinWidth end

"""
    MinFill
"""
struct MinFill end

"""
    InferenceProblem(wd::AbstractUWD, bm::AbstractDict)
"""
function InferenceProblem(wd::AbstractUWD, bm::AbstractDict)
    InferenceProblem{Int, Valuation{Int}}(wd, bm)
end

function InferenceProblem(wd::UntypedRelationDiagram{<:Any, T}, bm::AbstractDict) where T
    InferenceProblem{T, Valuation{T}}(wd, bm)
end

"""
    InferenceProblem{T₁, T₂}(wd::AbstractUWD, bm::AbstractDict) where {
        T₁, T₂ <: Valuation{T₁}}
"""
function InferenceProblem{T₁, T₂}(wd::AbstractUWD, bm::AbstractDict) where {
    T₁, T₂ <: Valuation{T₁}}
    bs = [bm[x] for x in subpart(wd, :name)]
    InferenceProblem{T₁, T₂}(wd, bs)
end

"""
    InferenceProblem(wd::AbstractUWD, bs)

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
function InferenceProblem(wd::AbstractUWD, bs)
    InferenceProblem{Int, Valuation{Int}}(wd, bs)
end

"""
    InferenceProblem{T₁, T₂}(wd::AbstractUWD, bs) where {
        T₁, T₂ <: Valuation{T₁}}
"""
function InferenceProblem{T₁, T₂}(wd::AbstractUWD, bs) where {
    T₁, T₂ <: Valuation{T₁}}
    @assert nboxes(wd) == length(bs)
    ls = [T₁[] for box in bs]
    for i in ports(wd; outer=false)
        push!(ls[box(wd, i)], junction(wd, i; outer=false))
    end
    query = [
        junction(wd, i; outer=true)
        for i in ports(wd; outer=true)]
    kb = [
        UWDBox(labels, box, false)
        for (labels, box) in zip(ls, bs)]
    InferenceProblem{T₁, T₂}(query, kb)
end

function InferenceProblem{T₁, T₂}(wd::UntypedRelationDiagram, bs) where {
    T₁, T₂ <: Valuation{T₁}}
    @assert nboxes(wd) == length(bs)
    ls = [T₁[] for box in bs]
    for i in ports(wd; outer=false)
        push!(ls[box(wd, i)], subpart(wd, junction(wd, i; outer=false), :variable))
    end
    query = [
        subpart(wd, junction(wd, i; outer=true), :variable)
        for i in ports(wd; outer=true)]
    kb = [
        UWDBox(labels, box, false)
        for (labels, box) in zip(ls, bs)]
    InferenceProblem{T₁, T₂}(query, kb)
end

"""
    init(ip::InferenceProblem, alg)
"""
init(ip::InferenceProblem, alg)

function init(ip::InferenceProblem{T₁, T₂}, ::MinWidth) where {T₁, T₂}
    order = minwidth!(primalgraph(ip.kb), ip.query)
    JoinTree{T₁, T₂}(ip.kb, order)
end

function init(ip::InferenceProblem{T₁, T₂}, ::MinFill) where {T₁, T₂}
    order = minfill!(primalgraph(ip.kb), ip.query)
    JoinTree{T₁, T₂}(ip.kb, order)
end

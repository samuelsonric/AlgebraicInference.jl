"""
    InferenceProblem{T₁, T₂ <: Valuation{T₁}}
"""
mutable struct InferenceProblem{T₁, T₂ <: Valuation{T₁}}
    query::Vector{T₁}
    kb::Vector{<:T₂}

    function InferenceProblem{T₁, T₂}(query, kb::Vector{<:T₂}) where {
        T₁, T₂ <: Valuation{T₁}}
        new{T₁, T₂}(query, kb)
    end
end

"""
    MinWidth

The min-width elimination heuristic.
"""
struct MinWidth end

"""
    MinFill

The min-fill elimination heuristic.
"""
struct MinFill end

"""
    InferenceProblem{T₁, T₂}(query, kb) where {T₁, T₂ <: Valuation{T₁}}
"""
function InferenceProblem{T₁, T₂}(query, kb) where {T₁, T₂ <: Valuation{T₁}}
    kb = map(ϕ -> convert(T₂, ϕ), kb)
    InferenceProblem{T₁, T₂}(query, kb)
end

"""
    InferenceProblem(query, kb)
"""
function InferenceProblem(query, kb)
    T = eltype(query)
    InferenceProblem{T, Valuation{T}}(query, kb)
end

"""
    InferenceProblem(wd::AbstractUWD, bm::AbstractDict)
"""
function InferenceProblem(wd::AbstractUWD, bm::AbstractDict)
    T = vtype(wd)
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
"""
function InferenceProblem(wd::AbstractUWD, bs)
    T = vtype(wd)
    InferenceProblem{T, Valuation{T}}(wd, bs)
end

"""
    InferenceProblem{T₁, T₂}(wd::AbstractUWD, bs) where {
        T₁, T₂ <: Valuation{T₁}}
"""
function InferenceProblem{T₁, T₂}(wd::AbstractUWD, bs) where {
    T₁, T₂ <: Valuation{T₁}}
    InferenceProblem{T₁, T₂}(wd, collect(bs))
end

function InferenceProblem{T₁, T₂}(wd::AbstractUWD, bs::AbstractVector) where {
    T₁, T₂ <: Valuation{T₁}}
    @assert nboxes(wd) == length(bs)
    ls = [T₁[] for box in bs]
    vs = T₁[]
    for i in ports(wd; outer=false)::UnitRange{Int}
        j = junction(wd, i; outer=false)
        push!(ls[box(wd, i)], j)
        push!(vs, j)
    end
    query = T₁[
        junction(wd, i; outer=true)
        for i in ports(wd; outer=true)::UnitRange{Int}]
    kb = [
        convert(T₂, UWDBox(labels, box, false))
        for (labels, box) in zip(ls, bs)]
    l = setdiff(query, vs)
    if !isempty(l)
        push!(kb, one(T₂, l))
    end
    InferenceProblem{T₁, T₂}(query, kb)
end

function InferenceProblem{T₁, T₂}(wd::RelationDiagram, bs::AbstractVector) where {
    T₁, T₂ <: Valuation{T₁}}
    @assert nboxes(wd) == length(bs)
    ls = [T₁[] for box in bs]
    for i in ports(wd; outer=false)::UnitRange{Int}
        push!(ls[box(wd, i)], subpart(wd, junction(wd, i; outer=false), :variable))
    end
    query = T₁[
        subpart(wd, junction(wd, i; outer=true), :variable)
        for i in ports(wd; outer=true)::UnitRange{Int}]
    kb = [
        convert(T₂, UWDBox(labels, box, false))
        for (labels, box) in zip(ls, bs)]
    InferenceProblem{T₁, T₂}(query, kb)
end

"""
    init(ip::InferenceProblem, alg)

Construct a covering join tree using variable elimination.

The argument `alg` specifies an elimination heuristic. Options are
- [`MinWidth()`](@ref)
- [`MinFill()`](@ref)
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

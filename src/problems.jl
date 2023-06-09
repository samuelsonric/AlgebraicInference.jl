"""
    InferenceProblem{T₁, T₂ <: Valuation{T₁}}
"""
mutable struct InferenceProblem{T₁, T₂ <: Valuation{T₁}}
    query::Vector{T₁}
    kb::Vector{T₂}

    """
        InferenceProblem{T₁, T₂}(query, kb)
    """
    function InferenceProblem{T₁, T₂}(query, kb) where {T₁, T₂ <: Valuation{T₁}}
        new{T₁, T₂}(query, kb)
    end
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
    InferenceProblem(query, kb)
"""
function InferenceProblem(query, kb)
    T₁ = eltype(query)
    T₂ = eltype(kb)
    InferenceProblem{T₁, T₂}(query, kb)
end

"""
    InferenceProblem(wd::UndirectedWiringDiagram, box_map::AbstractDict)
"""
function InferenceProblem(
    wd::UndirectedWiringDiagram,
    box_map::AbstractDict{<:Any, T}) where T

    InferenceProblem{Int, LabeledBox{Int, T}}(wd, box_map)
end

function InferenceProblem(
    wd::UntypedRelationDiagram{<:Any, T₁},
    box_map::AbstractDict{<:Any, T₂}) where {T₁, T₂}

    InferenceProblem{T₁, LabeledBox{T₁, T₂}}(wd, box_map)
end

"""
    InferenceProblem(wd::UndirectedWiringDiagram, boxes)

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
function InferenceProblem(wd::UndirectedWiringDiagram, boxes)
    T = eltype(boxes)
    InferenceProblem{Int, LabeledBox{Int, T}}(wd, boxes)
end

"""
    InferenceProblem{T₁, LabeledBox{T₁, T₂}}(
        wd::UndirectedWiringDiagram,
        box_map::AbstractDict) where {T₁, T₂}
"""
function InferenceProblem{T₁, LabeledBox{T₁, T₂}}(
    wd::UndirectedWiringDiagram,
    box_map::AbstractDict) where {T₁, T₂}

    boxes = T₂[box_map[x] for x in subpart(wd, :name)]
    InferenceProblem{T₁, LabeledBox{T₁, T₂}}(wd, boxes)
end

"""
    InferenceProblem{T₁, LabeledBox{T₁, T₂}}(
        wd::UndirectedWiringDiagram,
        boxes) where {T₁, T₂}
"""
function InferenceProblem{T₁, LabeledBox{T₁, T₂}}(
    wd::UndirectedWiringDiagram,
    boxes) where {T₁, T₂}

    InferenceProblem{T₁, LabeledBox{T₁, T₂}}(wd, collect(boxes))
end

function InferenceProblem{T₁, LabeledBox{T₁, T₂}}(
    wd::UndirectedWiringDiagram,
    boxes::AbstractVector) where {T₁, T₂}

    @assert nboxes(wd) == length(boxes)
    kb_labels = [T₁[] for box in boxes]
    for i in ports(wd; outer=false)
        push!(kb_labels[box(wd, i)], junction(wd, i; outer=false))
    end
    query = [
        junction(wd, i; outer=true)
        for i in ports(wd; outer=true)]
    kb = [
        LabeledBox{T₁, T₂}(labels, box)
        for (labels, box) in zip(kb_labels, boxes)]
    InferenceProblem{T₁, LabeledBox{T₁, T₂}}(query, kb)
end

function InferenceProblem{T₁, LabeledBox{T₁, T₂}}(
    wd::UntypedRelationDiagram,
    boxes::AbstractVector) where {T₁, T₂}

    @assert nboxes(wd) == length(boxes)
    kb_labels = [T₁[] for box in boxes]
    for i in ports(wd; outer=false)
        push!(kb_labels[box(wd, i)], subpart(wd, junction(wd, i; outer=false), :variable))
    end
    query = [
        subpart(wd, junction(wd, i; outer=true), :variable)
        for i in ports(wd; outer=true)]
    kb = [
        LabeledBox{T₁, T₂}(labels, box)
        for (labels, box) in zip(kb_labels, boxes)]
    InferenceProblem{T₁, LabeledBox{T₁, T₂}}(query, kb)
end

"""
    init(prob::InferenceProblem, alg)
"""
init(prob::InferenceProblem, alg)

function init(prob::InferenceProblem{T₁, T₂}, ::MinWidth) where {T₁, T₂}
    JoinTree{T₁, T₂}(prob.kb, minwidth!(primal_graph(prob.kb), prob.query))
end

function init(prob::InferenceProblem{T₁, T₂}, ::MinFill) where {T₁, T₂}
    JoinTree{T₁, T₂}(prob.kb, minfill!(primal_graph(prob.kb), prob.query))
end

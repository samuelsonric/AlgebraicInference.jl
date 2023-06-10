"""
    InferenceProblem{T₁ <: Valuation, T₂}
"""
mutable struct InferenceProblem{T₁ <: Valuation, T₂}
    kb::Vector{<:T₁}
    query::Vector{T₂}

    function InferenceProblem{T₁, T₂}(kb::Vector{<:T₁}, query) where {T₁, T₂}
        new{T₁, T₂}(kb, query)
    end
end

"""
    UWDProblem{T₁, T₂} = InferenceProblem{UWDBox{T₁, T₂}, T₂}
"""
const UWDProblem{T₁, T₂} = InferenceProblem{UWDBox{T₁, T₂}, T₂}

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
    InferenceProblem{T₁, T₂}(kb, query) where {T₁, T₂}
"""
function InferenceProblem{T₁, T₂}(kb, query) where {T₁, T₂}
    kb = map(ϕ -> convert(T₁, ϕ), kb)
    InferenceProblem{T₁, T₂}(kb, query)
end

"""
    UWDProblem{T}(wd::AbstractUWD, bm::AbstractDict) where T
"""
function UWDProblem{T}(wd::AbstractUWD, bm::AbstractDict) where T
    bs = [bm[x] for x in subpart(wd, :name)]
    UWDProblem{T}(wd, bs)
end

"""
    UWDProblem{T}(wd::AbstractUWD, bs) where T

Translate an undirected wiring diagram into an inference problem over a valuation algebra.
"""
function UWDProblem{T}(wd::AbstractUWD, bs) where T
    UWDProblem{T}(wd, collect(bs))
end

function UWDProblem{T}(wd::AbstractUWD, bs::Vector) where T
    @assert nboxes(wd) == length(bs)
    ls = [Int[] for box in bs]
    vs = Int[]
    for i in ports(wd; outer=false)::UnitRange{Int}
        v = junction(wd, i; outer=false)
        push!(ls[box(wd, i)], v)
        push!(vs, v)
    end
    query = [
        junction(wd, i; outer=true)
        for i in ports(wd; outer=true)::UnitRange{Int}]
    kb = [
        UWDBox{T, Int}(box, labels, false)
        for (labels, box) in zip(ls, bs)]
    push!(kb, one(UWDBox{T, Int}, setdiff(query, vs)))
    UWDProblem{T, Int}(kb, query)
end

function UWDProblem{T₁}(wd::UntypedRelationDiagram{<:Any, T₂}, bs::Vector) where {T₁, T₂}
    @assert nboxes(wd) == length(bs)
    ls = [T₂[] for box in bs]
    vs = T₂[]
    for i in ports(wd; outer=false)::UnitRange{Int}
        v = subpart(wd, junction(wd, i; outer=false), :variable)
        push!(ls[box(wd, i)], v)
        push!(vs, v)
    end
    query = [
        subpart(wd, junction(wd, i; outer=true), :variable)
        for i in ports(wd; outer=true)::UnitRange{Int}]
    kb = [
        UWDBox{T₁, T₂}(box, labels, false)
        for (labels, box) in zip(ls, bs)]
    push!(kb, one(UWDBox{T₁, T₂}, setdiff(query, vs)))
    UWDProblem{T₁, T₂}(kb, query)
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
    jt = JoinTree{T₁, T₂}(ip.kb, order)
    InferenceSolver{T₁, T₂}(jt, ip.query)
end

function init(ip::InferenceProblem{T₁, T₂}, ::MinFill) where {T₁, T₂}
    order = minfill!(primalgraph(ip.kb), ip.query)
    jt = JoinTree{T₁, T₂}(ip.kb, order)
    InferenceSolver{T₁, T₂}(jt, ip.query)
end

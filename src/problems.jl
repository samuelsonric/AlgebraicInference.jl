"""
    InferenceProblem{T₁ <: Valuation, T₂}

An inference problem over a valuation algebra. Construct a solver for an inference problem
with the function [`init`](@ref), or solve it directly with [`solve`](@ref).
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

An inference problem that performs undirected composition.
"""
const UWDProblem{T₁, T₂} = InferenceProblem{UWDBox{T₁, T₂}, T₂}

"""
    MinWidth

Contructs a covering join tree for an inference problem using the variable elimination
algorithm. Variables are eliminated according to the "min-width" heuristic.
"""
struct MinWidth end

"""
    MinFill

Contructs a covering join tree for an inference problem using the variable elimination
algorithm. Variables are eliminated according to the "min-fill" heuristic.
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
    UWDProblem{T}(wd::AbstractUWD, bm::AbstractDict, jm=nothing;
        ha=:name, ja=:name) where T

Construct an inference problem that performs undirected composition. Before being composed,
the values of `bm` are converted to type `T`.
"""
function UWDProblem{T}(wd::AbstractUWD, bm::AbstractDict, jm=nothing;
    ha=:name, ja=:variable) where T
    bs = [bm[x] for x in subpart(wd, ha)]
    js = isnothing(jm) ? nothing : [jm[x] for x in subpart(wd, ja)]
    UWDProblem{T}(wd, bs, js)
end

"""
    UWDProblem{T}(wd::AbstractUWD, bs, js=nothing) where T

Construct an inference problem that performs undirected composition. Before being composed,
the elements of `bs` are converted to type `T`.
"""
function UWDProblem{T}(wd::AbstractUWD, bs, js=nothing) where T
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
    rv = setdiff(query, vs)
    if !isempty(rv)
        js = isnothing(js) ? nothing : [js[i] for i in rv]
        push!(kb, one(UWDBox{T, Int}, rv, js))
    end
    UWDProblem{T, Int}(kb, query)
end

"""
    solve(ip::InferenceProblem, alg)

Solve an inference problem. The options for `alg` are
- [`MinWidth()`](@ref)
- [`MinFill()`](@ref)
"""
solve(ip::InferenceProblem, alg)

"""
    init(ip::InferenceProblem, alg)

Construct a solver for an inference problem. The options for `alg` are
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

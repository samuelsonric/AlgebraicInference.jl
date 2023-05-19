"""
    osla_sc(hyperedges::Vector{Set{T}}, vertices::Set{T}) where T

Let ``(V, E)`` be a hypergraph and ``S \\subseteq V`` a collection of vertices. Then
`olsa_sc(hyperedges, vertices)` constructs an ordering of ``S`` using the "One Step Look
Ahead - Smallest Clique" heuristic.

References:
- Lehmann, N. 2001. *Argumentation System and Belief Functions*. Ph.D. thesis, Department
  of Informatics, University of Fribourg.
"""
function osla_sc(hyperedges::Vector{Set{T}}, vertices::Set{T}) where T
    hyperedges = copy(hyperedges); vertices = copy(vertices)
    elimination_sequence = T[]
    while !isempty(vertices)
        X = mask = cl = nothing
        for _X in vertices
            _mask = [_X in s for s in hyperedges]
            _cl = ∪(hyperedges[_mask]...)
            if sum(_mask) <= 1
                X = _X; mask = _mask; cl = _cl
                break
            end
            if isnothing(X) || length(_cl) < length(cl)
                X = _X; mask = _mask; cl = _cl
            end
        end
        keepat!(hyperedges, .!mask); push!(hyperedges, setdiff(cl, [X]))
        push!(elimination_sequence, X) 
        delete!(vertices, X)
    end
    elimination_sequence
end

function primal_graph(hyperedges::Vector{Set{T}}) where T
    edges = Set{T}[]
    for s in hyperedges
        for X₁ in s
            for X₂ in s
                if !isequal(X₁, X₂)
                    push!(edges, Set([X₁, X₂]))
                end
            end
        end
    end
    edges
end

function osla_ffi(edges::Vector{Set{T}}, vertices::Set{T}) where T
    edges = copy(edges); vertices = copy(vertices)
    elimination_sequence = T[]
    while !isempty(vertices)
        X = mask = fi = nothing
        for _X in vertices
            _mask = [_X in s for s in edges]
            _N = [first(setdiff(s, [_X])) for s in edges[_mask]]
            _fi = [Set([Y₁, Y₂]) for Y₁ in _N for Y₂ in _N if !isequal(Y₁, Y₂)]
            setdiff!(_fi, edges[.!_mask])
            if isnothing(X) || length(_fi) < length(fi)
                X = _X; mask = _mask; fi = _fi
            end
        end
        keepat!(edges, mask); append!(edges, fi)
        push!(elimination_sequence, X)
        delete!(vertices, X)
    end
    elimination_sequence, edges
end

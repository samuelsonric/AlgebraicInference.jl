function primal_graph(hyperedges::Vector{Set{T}}) where T
    edges = Set{Set{T}}()
    for s in hyperedges
        for X₁ in s
            for X₂ in s
                if !isequal(X₁, X₂)
                    push!(edges, Set([X₁, X₂]))
                end
            end
        end
    end
    collect(edges)
end

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
        X = mask = ne = nothing
        for _X in vertices
            _mask = [_X in s for s in hyperedges]
            _ne = Set{T}(); union!(_ne, hyperedges[_mask]...); delete!(_ne, _X)
            if sum(_mask) <= 1
                X = _X; mask = _mask; ne = _ne
                break
            end
            if isnothing(X) || length(_ne) < length(ne)
                X = _X; mask = _mask; ne = _ne
            end
        end
        keepat!(hyperedges, .!mask); push!(hyperedges, ne)
        push!(elimination_sequence, X)
        delete!(vertices, X)
    end
    elimination_sequence
end

function osla_ffi(edges::Vector{Set{T}}, vertices::Set{T}) where T
    edges = copy(edges); vertices = copy(vertices)
    elimination_sequence = T[]
    while !isempty(vertices)
        X = mask = fi = nothing
        for _X in vertices
            _mask = [_X in s for s in edges]
            _ne = Set{T}(); union!(_ne, edges[_mask]...); delete!(_ne, _X)
            _fi = primal_graph([_ne]); setdiff!(_fi, edges[.!_mask])
            if isnothing(X) || length(_fi) < length(fi)
                X = _X; mask = _mask; fi = _fi
            end
        end
        keepat!(edges, .!mask); append!(edges, fi)
        push!(elimination_sequence, X)
        delete!(vertices, X)
    end
    elimination_sequence
end

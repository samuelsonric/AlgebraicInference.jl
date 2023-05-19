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
        push!(elimination_sequence, X); delete!(cl, X)
        keepat!(hyperedges, .!mask); push!(hyperedges, cl)
        delete!(vertices, X)
    end
    elimination_sequence
end

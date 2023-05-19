"""
    construct_elimination_sequence(edges::AbstractVector{T₂},
                                   query::AbstractSet) where {T₁, T₂ <: AbstractSet{T₁}}

Construct an elimination sequence using the "One Step Look Ahead - Smallest Clique"
heuristic.

Let ``(V, E)`` be a hypergraph and ``x \\subseteq V`` a query. Then
`construct_elimination_sequence(edges, query)` constructs an ordering ``(X_1, \\dots, X_m)``
of the vertices in ``V - x``.

References:
- Lehmann, N. 2001. *Argumentation System and Belief Functions*. Ph.D. thesis, Department
  of Informatics, University of Fribourg.
"""
function osla_sc(hyperedges::Vector{Set{T}}, variables::Set{T}) where T
    hyperedges = copy(hyperedges); variables = copy(variables)
    elimination_sequence = T[]
    while !isempty(variables)
        X = mask = cl = nothing
        for _X in variables
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
        delete!(variables, X)
    end
    elimination_sequence
end

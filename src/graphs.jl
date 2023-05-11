"""
    construct_elimination_sequence(edges::AbstractSet{<:AbstractSet{<:Variable}},
                                   query::AbstractSet{<:Variable})

Construct an elimination sequence using the "One Step Look Ahead - Smallest Clique"
heuristic.

Let ``(V, E)`` be a hypergraph and ``x \\subseteq V`` a query. Then
`construct_elimination_sequence(edges, query)` constructs an ordering ``(X_1, \\dots, X_m)``
of the vertices in ``V - x``.

References:
- Lehmann, N. 2001. *Argumentation System and Belief Functions*. Ph.D. thesis, Department
  of Informatics, University of Fribourg.
"""
function construct_elimination_sequence(domains::AbstractSet{<:AbstractSet},
                                        query::AbstractSet)
    E = domains; x = query
    Xs = setdiff(∪(E...), x)
    if isempty(Xs)
        return []
    else
        X = argmin(Xs) do X
            Eₓ = Set(s for s in E if X in s)
            length(∪(Eₓ...))
        end
        Eₓ = Set(s for s in E if X in s)
        sₓ = ∪(Eₓ...)
        F = setdiff(E, Eₓ) ∪ [setdiff(sₓ, [X])]
        return [X, construct_elimination_sequence(F, x)...]
    end
end


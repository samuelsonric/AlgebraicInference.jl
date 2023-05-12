struct LabeledGraph{T}
    labels::Vector{Set{T}}
    edges::Set{Set{Int}}
end

function child(tree::LabeledGraph, i::Integer)
    for j in i + 1:length(tree.labels)
        if Set([i, j]) in tree.edges
            return j
        end
    end
    error()
end

"""
    construct_elimination_sequence(domains::Set{Set{T}},
                                   query::AbstractSet) where T

Construct an elimination sequence using the "One Step Look Ahead - Smallest Clique"
heuristic.

Let ``(V, E)`` be a hypergraph and ``x \\subseteq V`` a query. Then
`construct_elimination_sequence(edges, query)` constructs an ordering ``(X_1, \\dots, X_m)``
of the vertices in ``V - x``.

References:
- Lehmann, N. 2001. *Argumentation System and Belief Functions*. Ph.D. thesis, Department
  of Informatics, University of Fribourg.
"""
function construct_elimination_sequence(domains::Set{Set{T}},
                                        query::AbstractSet) where T
    elimination_sequence = T[]
    E = copy(domains); x = query
    V = setdiff(∪(E...), x)
    while !(isempty(V))
        X = argmin(V) do X
            Eₓ = Set(s for s in E if X in s)
            length(∪(Eₓ...))
        end
        push!(elimination_sequence, X)
        Eₓ = Set(s for s in E if X in s)
        sₓ = ∪(Eₓ...)
        setdiff!(E, Eₓ); push!(E, setdiff(sₓ, [X]))
        V = setdiff(∪(E...), x)
    end
    elimination_sequence
end

"""
    construct_join_tree(domains::Set{Set{T}},
                        elimination_sequence::AbstractVector) where T
"""
function construct_join_tree(domains::Set{Set{T}},
                             elimination_sequence::AbstractVector) where T
    λ = Set{T}[]; color = Bool[]
    V = 0; E = Set{Set{Int}}()
    l = domains
    for X in elimination_sequence
        lₓ = Set(s for s in l if X in s )
        sₓ = ∪(lₓ...)
        l = setdiff(l, lₓ) ∪ [setdiff(sₓ, [X])]
        i = V + 1; push!(λ, sₓ); push!(color, true)
        for j in 1:V
            if X in λ[j] && color[j]
                push!(E, Set([i, j]))
                color[j] = false
            end
        end
        V += 1
    end
    i = V + 1; push!(λ, ∪(l...))
    for j in 1:V
        if color[j]
            push!(E, Set([i, j]))
            color[j] = false
        end
    end
    LabeledGraph(λ, E)
end

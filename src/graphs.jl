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
function construct_elimination_sequence(edges::AbstractVector{T₂},
                                        query::AbstractSet) where {T₁, T₂ <: AbstractSet{T₁}}
    fused_edges = T₂[edges...]
    elimination_sequence = T₁[]
    variables = setdiff(∪(fused_edges...), query)
    while !(isempty(variables))
        X = argmin(variables) do X
            mask = [X in s for s in fused_edges]
            edge = ∪(fused_edges[mask]...)
            length(edge)
        end
        mask = [X in s for s in fused_edges]
        edge = ∪(fused_edges[mask]...)
        push!(elimination_sequence, X)
        keepat!(fused_edges, .!mask); push!(fused_edges, setdiff(edge, [X]))
        variables = setdiff(∪(fused_edges...), query)
    end
    elimination_sequence
end

"""
    construct_join_tree(edges::AbstractVector{T₂},
                        elimination_sequence) where {T₁, T₂ <: AbstractSet{T₁}}
"""
function construct_join_tree(edges::AbstractVector{T₂},
                             elimination_sequence) where {T₁, T₂ <: AbstractSet{T₁}}
    fused_edges = T₂[edges...]
    labels = T₂[]; color = Bool[]; vertices = Node{Int}[]
    for X in elimination_sequence
        mask = [X in s for s in fused_edges]
        edge = ∪(fused_edges[mask]...)
        keepat!(fused_edges, .!mask); push!(fused_edges, setdiff(edge, [X]))
        push!(labels, edge); push!(color, true)
        i = Node(length(vertices) + 1)
        for j in vertices
            if X in labels[j.id] && color[j.id]
                push!(i.children, j)
                j.parent = i
                color[j.id] = false
            end
        end
        push!(vertices, i)
    end
    push!(labels, ∪(fused_edges...))
    tree = Node(length(vertices) + 1)
    for j in vertices
        if color[j.id]
            push!(tree.children, j)
            j.parent = tree
            color[j.id] = false
        end
    end
    labels, tree
end

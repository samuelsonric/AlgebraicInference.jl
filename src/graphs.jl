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
function construct_elimination_sequence(domains::AbstractSet{T₂},
                                        query::AbstractSet) where {T₁, T₂ <: AbstractSet{T₁}}
    edges = Set(domains)
    elimination_sequence = T₁[]
    variables = setdiff(∪(edges...), query)
    while !(isempty(variables))
        X = argmin(variables) do X
            E = Set(s for s in edges if X in s)
            length(∪(E...))
        end
        E = Set(s for s in edges if X in s)
        s = ∪(E...)
        push!(elimination_sequence, X)
        setdiff!(edges, E); push!(edges, setdiff(s, [X]))
        variables = setdiff(∪(edges...), query)
    end
    elimination_sequence
end

"""
    construct_join_tree(domains::Set{Set{T}},
                        elimination_sequence::AbstractVector) where T
"""
function construct_join_tree(knowledge_base_domains::AbstractSet{T₂},
                             elimination_sequence::AbstractVector) where {T₁, T₂ <: AbstractSet{T₁}}
    knowledge_base_domains = Set(knowledge_base_domains)
    join_tree_domains = T₂[]; color = Bool[]; vertices = Node{Int}[]
    for X in elimination_sequence
        domains = Set(s for s in knowledge_base_domains if X in s)
        fused_domain = ∪(domains...)
        setdiff!(knowledge_base_domains, domains)
        push!(knowledge_base_domains, setdiff(fused_domain, [X]))
        push!(join_tree_domains, fused_domain); push!(color, true)
        i = Node(length(vertices) + 1)
        for j in vertices
            if X in join_tree_domains[j.id] && color[j.id]
                push!(i.children, j)
                j.parent = i
                color[j.id] = false
            end
        end
        push!(vertices, i)
    end
    push!(join_tree_domains, ∪(knowledge_base_domains...))
    join_tree = Node(length(vertices) + 1)
    for j in vertices
        if color[j.id]
            push!(join_tree.children, j)
            j.parent = join_tree
            color[j.id] = false
        end
    end
    join_tree_domains, join_tree
end

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
function construct_elimination_sequence(domains::AbstractVector{T₂},
                                        query::AbstractSet) where {T₁, T₂ <: AbstractSet{T₁}}
    domains = T₂[domains...]
    elimination_sequence = T₁[]
    variables = setdiff(∪(domains...), query)
    while !(isempty(variables))
        X = argmin(variables) do X
            mask = [X in s for s in domains]
            domain = ∪(domains[mask]...)
            length(domain)
        end
        mask = [X in s for s in domains]
        domain = ∪(domains[mask]...)
        push!(elimination_sequence, X)
        keepat!(domains, .!mask); push!(domains, setdiff(domain, [X]))
        variables = setdiff(∪(domains...), query)
    end
    elimination_sequence
end

"""
    construct_join_tree(domains::Set{Set{T}},
                        elimination_sequence::AbstractVector) where T
"""
function construct_join_tree(domains::AbstractVector{T₂},
                             elimination_sequence::AbstractVector) where {T₁, T₂ <: AbstractSet{T₁}}
    domains = T₂[domains...]
    join_tree_domains = T₂[]; color = Bool[]; vertices = Node{Int}[]
    for X in elimination_sequence
        mask = [X in s for s in domains]
        domain = ∪(domains[mask]...)
        keepat!(domains, .!mask); push!(domains, setdiff(domain, [X]))
        push!(join_tree_domains, domain); push!(color, true)
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
    push!(join_tree_domains, ∪(domains...))
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

function ChildIndexing(::Type{<:JoinTree})
    IndexedChildren()
end

function NodeType(::Type{<:JoinTree})
    HasNodeType()
end

function ParentLinks(::Type{<:JoinTree})
    StoredParents()
end

function children(node::JoinTree)
    node.children
end

function nodetype(::Type{T}) where T <: JoinTree
    T
end

function nodevalue(node::JoinTree)
    node.id
end

function parent(node::JoinTree)
    node.parent
end

function message_to_parent(node::JoinTree)
    @assert !isroot(node)
    factor = node.factor
    for child in node.children
        factor = combine(factor, message_to_parent(child))
    end
    project(factor, domain(factor) ∩ node.parent.domain)
end

function message_to_parent!(node::JoinTree)
    @assert !isroot(node)
    if isnothing(node.message_to_parent)
        factor = node.factor
        for child in node.children
            factor = combine(factor, message_to_parent!(child))
        end
        node.message_to_parent = project(factor, domain(factor) ∩ node.parent.domain)
    end
    node.message_to_parent
end

function message_from_parent!(node::JoinTree)
    @assert !isroot(node)
    if isnothing(node.message_from_parent)
        factor = node.factor
        for sibling in node.parent.children
            if node.id != sibling.id
                factor = combine(factor, message_to_parent!(sibling))
            end
        end
        if !isroot(node.parent)
            factor = combine(factor, message_from_parent!(node.parent))
        end
        node.message_from_parent = project(factor, domain(factor) ∩ node.domain)
    end
    node.message_from_parent
end

function construct_factors!(join_tree::JoinTree{T₁, T₂},
                            assignment_map::Vector{T₁},
                            knowledge_base::Vector{<:Valuation{T₂}};
                            identity=true) where {T₁, T₂}
    node_map = Dict(node.id => node for node in PreOrderDFS(join_tree))
    e = IdentityValuation{T₂}()
    for node in values(node_map)
        node.factor = identity ? e : neutral_valuation(node.domain)
    end
    for (i, j) in enumerate(assignment_map)
        node_map[j].factor = combine(node_map[j].factor, knowledge_base[i])
    end
end

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

"""
    construct_join_tree(edges::AbstractVector{T₂},
                        elimination_sequence) where {T₁, T₂ <: AbstractSet{T₁}}
"""
function construct_join_tree(hyperedges::Vector{Set{T}}, elimination_sequence::Vector{T}) where T
    hyperedges = copy(hyperedges)
    color = Bool[]; nodes = JoinTree{Int, T}[]
    for X in elimination_sequence
        mask = [X in s for s in hyperedges]
        cl = ∪(hyperedges[mask]...)
        keepat!(hyperedges, .!mask); push!(hyperedges, setdiff(cl, [X]))
        i = JoinTree(length(nodes) + 1, cl); push!(color, true)
        for j in nodes
            if X in j.domain && color[j.id]
                push!(i.children, j)
                j.parent = i
                color[j.id] = false
            end
        end
        push!(nodes, i)
    end
    tree = JoinTree(length(nodes) + 1, ∪(hyperedges...))
    for j in nodes
        if color[j.id]
            push!(tree.children, j)
            j.parent = tree
            color[j.id] = false
        end
    end
    tree
end

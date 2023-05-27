"""
    Architecture{T₁, T₂} <: AbstractNode{T₁}

An `Architecture` contains a join tree ``(V, E, \\lambda, D)``. To each vertex ``i \\in V``
it optionally associates a factor ``\\psi_i`` and mailbox
``(\\mu_{i \\to pa(i)}, \\mu_{pa(i) \\to i})``.
"""
mutable struct Architecture{T₁, T₂} <: AbstractNode{T₁}
    id::T₁
    domain::Vector{T₂}
    children::Vector{Architecture{T₁, T₂}}
    parent::Union{Nothing, Architecture{T₁, T₂}}
    factor::Union{Nothing, Valuation{T₂}}
    message_from_parent::Union{Nothing, Valuation{T₂}}
    message_to_parent::Union{Nothing, Valuation{T₂}}

    function Architecture(id::T₁, domain::Vector{T₂}) where {T₁, T₂}
        new{T₁, T₂}(id, domain, Architecture{T₁, T₂}[], nothing, nothing, nothing, nothing)
    end
end

function ChildIndexing(::Type{<:Architecture})
    IndexedChildren()
end

function NodeType(::Type{<:Architecture})
    HasNodeType()
end

function ParentLinks(::Type{<:Architecture})
    StoredParents()
end

function children(node::Architecture)
    node.children
end

function nodetype(::Type{T}) where T <: Architecture
    T
end

function nodevalue(node::Architecture)
    node.id
end

function parent(node::Architecture)
    node.parent
end

#=
"""
    function construct_join_tree(hyperedges::Vector{Set{T}},
                                 elimination_sequence::Vector{T}) where T <: Variable

Construct a join tree by eliminating the variables in `elimination_sequence`.
"""
function construct_join_tree(hyperedges::Vector{Set{T}},
                             elimination_sequence::Vector{T}) where T <: Variable
    hyperedges = copy(hyperedges)
    color = Bool[]; nodes = Architecture{Int, T}[]
    for X in elimination_sequence
        mask = [X in s for s in hyperedges]
        cl = ∪(hyperedges[mask]...)
        keepat!(hyperedges, .!mask); push!(hyperedges, setdiff(cl, [X]))
        i = Architecture(length(nodes) + 1, cl); push!(color, true)
        for j in nodes
            if X in j.domain && color[j.id]
                push!(i.children, j)
                j.parent = i
                color[j.id] = false
            end
        end
        push!(nodes, i)
    end
    join_tree = Architecture(length(nodes) + 1, ∪(hyperedges...))
    for j in nodes
        if color[j.id]
            push!(join_tree.children, j)
            j.parent = join_tree
            color[j.id] = false
        end
    end
    join_tree
end

"""
    function construct_factors!(architecture::Architecture{T₁, T₂},
                                assignment_map::Vector{T₁},
                                knowledge_base::Vector{<:Valuation{T₂}};
                                identity=true) where {T₁, T₂}

Let ``(V, E, \\lambda, D)`` be a join tree, ``\\{\\phi_1, \\dots, \\phi_n \\}`` a knowledge
base, and ``a: \\{1, \\dots, \\n\\} \\to V`` an assignment mapping. If `identity=true`,
then `construct_factors!(architecture, assignment_map, knowledge_base, identity)` assigns to
each vertex ``i \\in V`` the factor
```math
    \\psi_i = e \\otimes \\bigotimes_{j:a(j)=i} \\phi_j,
```
where ``e`` is the identity element. If `identity=false`, then the function uses neutral
elements instead of the identity element. 
"""
function construct_factors!(architecture::Architecture{T₁, T₂},
                            assignment_map::Vector{T₁},
                            knowledge_base::Vector{<:Valuation{T₂}};
                            identity=true) where {T₁, T₂}
    node_map = Dict(node.id => node for node in PreOrderDFS(architecture))
    e = IdentityValuation{T₂}()
    for node in values(node_map)
        node.factor = identity ? e : neutral_valuation(node.domain)
    end
    for (i, j) in enumerate(assignment_map)
        node_map[j].factor = combine(node_map[j].factor, knowledge_base[i])
    end
end
=#

"""
    answer_query(architecture::Architecture{T₁, T₂}, query::Set{T₂}) where {T₁, T₂}

Answer a query.

Let ``(V, E, \\lambda, D)`` be a join tree with factors ``\\{\\phi_i\\}_{i \\in V}``.
Let ``x`` be a query covered by ``(V, E, \\lambda, D)``. Then
`answer_query(architecture, query)` solves the inference problem
```math
\\left( \\bigotimes_{i \\in V} \\psi_i \\right)^{\\downarrow x}.
```
"""
function answer_query(architecture::Architecture, query::AbstractSet)
    for node in PreOrderDFS(architecture)
        if query ⊆ node.domain        
            factor = node.factor
            for child in node.children
                factor = combine(factor, message_to_parent(child))
            end
            if !isroot(node)
                factor = combine(factor, message_from_parent(node))
            end
            return project(factor, query)
        end 
    end
    error("Query not covered by join tree.")
end

"""
    answer_query!(architecture::Architecture{T₁, T₂}, query::Set{T₂}) where {T₁, T₂}

Answer a query, caching intermediate computations.

Let ``(V, E, \\lambda, D)`` be a join tree with factors ``\\{\\phi_i\\}_{i \\in V}``.
Let ``x`` be a query covered by ``(V, E, \\lambda, D)``. Then
`answer_query!(architecture, query)` solves the inference problem
```math
\\left( \\bigotimes_{i \\in V} \\psi_i \\right)^{\\downarrow x}.
```
"""
function answer_query!(architecture::Architecture, query::AbstractSet)
    for node in PreOrderDFS(architecture)
        if query ⊆ node.domain        
            factor = node.factor
            for child in node.children
                factor = combine(factor, message_to_parent!(child))
            end
            if !isroot(node)
                factor = combine(factor, message_from_parent!(node))
            end
            return project(factor, query)
        end 
    end
    error("Query not covered by join tree.")
end

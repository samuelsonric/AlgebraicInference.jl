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

function architecture(kb::Vector{<:Valuation{T}}, order) where T
    kb = copy(kb); pg = primal_graph(kb)
    color = Bool[]
    nodes = Architecture{Int, Int}[]
    e = IdentityValuation{T}()
    for X in order
        cl = collect(neighbor_labels(pg, X)); push!(cl, X)
        fa = e
        for i in length(kb):-1:1
            if X in domain(kb[i])
                fa = combine(fa, kb[i])
                deleteat!(kb, i)
            end
        end
        node = Architecture(length(nodes) + 1, cl); push!(color, true)
        node.factor = fa
        eliminate!(pg, code_for(pg, X))
        for _node in nodes
            if X in _node.domain && color[_node.id]
                push!(node.children, _node)
                _node.parent = node
                color[_node.id] = false
            end
        end
        push!(nodes, node)
    end
    node = Architecture(length(nodes) + 1, collect(labels(pg)))
    node.factor = reduce(combine, kb; init=e)
    for _node in nodes
        if color[_node.id]
            push!(node.children, _node)
            _node.parent = node
            color[_node.id] = false
        end
    end
    node
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
function answer_query(architecture::Architecture, query)
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
function answer_query!(architecture::Architecture, query)
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

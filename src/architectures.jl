"""
    Architecture{T₁, T₂ <: Valuation{T₁}} <: AbstractNode{Int}

A join tree ``(V, E, \\lambda, D)``, along with a set of factors
``\\left\\{ \\phi_i \\right\\}_{i \\in V}`` and mailboxes
```math
    \\left\\{ \\left( \\mu_{ i \\to \\mathtt{pa}(i)}, \\mu_{\\mathtt{pa}(i) \\to i} \\right) \\right\\}_{i \\in V}.
```
"""
mutable struct Architecture{T₁, T₂ <: Valuation{T₁}} <: AbstractNode{Int}
    id::Int
    domain::Vector{T₁}
    factor::T₂
    children::Vector{Architecture{T₁, T₂}}
    parent::Union{Nothing, Architecture{T₁, T₂}}
    message_from_parent::Union{Nothing, T₂}
    message_to_parent::Union{Nothing, T₂}

    function Architecture{T₁, T₂}(id, domain, factor) where {T₁, T₂ <: Valuation{T₁}}
        new{T₁, T₂}(id, domain, factor, Architecture{T₁, T₂}[], nothing, nothing, nothing)
    end
end

function Architecture{T₁, T₂}(kb, order) where {T₁, T₂ <: Valuation{T₁}}
    Architecture{T₁, T₂}(Vector{T₂}(kb), order) 
end

function Architecture{T₁, T₂}(kb::Vector{T₂}, order) where {T₁, T₂ <: Valuation{T₁}}
    kb = copy(kb)
    pg = primal_graph(kb)
 
    n = length(order)
    color = Vector{Bool}(undef, n)
    nodes = Vector{Architecture{T₁, T₂}}(undef, n)

    for i in 1:n
        X = order[i]
        ϕ = one(T₂)

        for j in length(kb):-1:1
            if X in domain(kb[j])
                ϕ = combine(ϕ, kb[j])
                deleteat!(kb, j)
            end
        end

        color[i] = true
        nodes[i] = Architecture{T₁, T₂}(i, [X, neighbor_labels(pg, X)...], ϕ)
        eliminate!(pg, code_for(pg, X))

        for j in 1:i-1
            if X in nodes[j].domain && color[j]
                color[j] = false
                nodes[j].parent = nodes[i]
                push!(nodes[i].children, nodes[j])
            end
        end
    end

    jt = Architecture{T₁, T₂}(n + 1, collect(labels(pg)), reduce(combine, kb; init=one(T₂)))

    for i in 1:n
        if color[i]
            nodes[i].parent = jt
            push!(jt.children, nodes[i])
        end
    end

    jt
end

function Architecture(id, domain, factor::Valuation{T}) where T
    Architecture{T, Valuation{T}}(id, domain, factor)
end

function Architecture(kb::AbstractVector{<:Valuation{T}}, order) where T
    Architecture{T, Valuation{T}}(kb, order)
end

"""
    architecture(kb::AbstractVector{<:Valuation{T}}, order) where T

Construct a covering join tree for the knowledge base `kb` using the variable elimination
order `order`.
"""
function architecture(kb::AbstractVector{<:Valuation{T}}, order) where T
    kb = copy(kb); pg = primal_graph(kb)
    color = Bool[]
    nodes = Architecture{T, Valuation{T}}[]
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
        node = Architecture(length(nodes) + 1, cl, fa); push!(color, true)
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
    node = Architecture(length(nodes) + 1, collect(labels(pg)), reduce(combine, kb; init=e))
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
    answer_query(jt::Architecture, query)

Answer a query.
"""
function answer_query(jt::T₂, query) where {T₁, T₂ <: Architecture{<:Any, T₁}}
    for node::T₂ in PreOrderDFS(jt)
        if query ⊆ node.domain        
            factor = node.factor
            for child in node.children
                factor = combine(factor, message_to_parent(child)::T₁)
            end
            if !isroot(node)
                factor = combine(factor, message_from_parent(node)::T₁)
            end
            return project(factor, query)
        end 
    end
    error("Query not covered by join tree.")
end

"""
    answer_query!(jt::Architecture, query)

Answer a query, caching intermediate computations in the mailboxes of `jt`.
"""
function answer_query!(jt::T₂, query) where {T₁, T₂ <: Architecture{<:Any, T₁}}
    for node::T₂ in PreOrderDFS(jt)
        if query ⊆ node.domain        
            factor = node.factor
            for child in node.children
                factor = combine(factor, message_to_parent!(child)::T₁)
            end
            if !isroot(node)
                factor = combine(factor, message_from_parent!(node)::T₁)
            end
            return project(factor, query)
        end 
    end
    error("Query not covered by join tree.")
end

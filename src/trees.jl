"""
    JoinTree{T₁, T₂ <: Valuation{T₁}} <: AbstractNode{Int}

A join tree with variables of type `T₁` and factors of type `T₂`.
"""
mutable struct JoinTree{T₁, T₂ <: Valuation{T₁}} <: AbstractNode{Int}
    id::Int
    domain::Vector{T₁}
    factor::T₂
    children::Vector{JoinTree{T₁, T₂}}
    parent::Union{Nothing, JoinTree{T₁, T₂}}
    message_from_parent::Union{Nothing, T₂}
    message_to_parent::Union{Nothing, T₂}

    @doc """
        JoinTree{T₁, T₂}(id, domain, factor) where {T₁, T₂ <: Valuation{T₁}}

    Construct a node in a join tree.
    """
    function JoinTree{T₁, T₂}(id, domain, factor) where {T₁, T₂ <: Valuation{T₁}}
        new{T₁, T₂}(id, domain, factor, JoinTree{T₁, T₂}[], nothing, nothing, nothing)
    end
end

"""
    JoinTree{T₁, T₂}(kb, order) where {T₁, T₂ <: Valuation{T₁}}

Construct a covering join tree for the knowledge base `kb` using the variable elimination
order `order`.

In order to use this function, you must define the method `one(::Type{T₂})`.
"""
function JoinTree{T₁, T₂}(kb, order) where {T₁, T₂ <: Valuation{T₁}}
    JoinTree{T₁, T₂}(map(ϕ -> convert(T₂, ϕ), kb), order) 
end

"""
    JoinTree(kb, order)

Construct a covering join tree for the knowledge base `kb` using the variable elimination
order `order`.
"""
function JoinTree(kb, order)
    JoinTree(collect(kb), order)
end

function JoinTree(kb::Vector{<:Valuation{T}}, order) where T
    JoinTree{T, Valuation{T}}(kb, order)
end

function JoinTree{T₁, T₂}(kb::Vector{<:T₂}, order) where {T₁, T₂ <: Valuation{T₁}}
    kb = copy(kb)
    pg = primal_graph(kb)
 
    n = length(order)
    color = Vector{Bool}(undef, n)
    nodes = Vector{JoinTree{T₁, T₂}}(undef, n)

    e = one(T₂)

    for i in 1:n
        X = order[i]
        ϕ = e

        for j in length(kb):-1:1
            if X in domain(kb[j])
                ϕ = combine(ϕ, kb[j])
                deleteat!(kb, j)
            end
        end

        color[i] = true
        nodes[i] = JoinTree{T₁, T₂}(i, [X, neighbor_labels(pg, X)...], ϕ)
        eliminate!(pg, code_for(pg, X))

        for j in 1:i - 1
            if X in nodes[j].domain && color[j]
                color[j] = false
                nodes[j].parent = nodes[i]
                push!(nodes[i].children, nodes[j])
            end
        end
    end

    jt = JoinTree{T₁, T₂}(n + 1, collect(labels(pg)), reduce(combine, kb; init=e))

    for i in 1:n
        if color[i]
            nodes[i].parent = jt
            push!(jt.children, nodes[i])
        end
    end

    jt
end

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

"""
    answer_query(jt::JoinTree, query)

Answer a query.
"""
function answer_query(jt::T₂, query) where {T₁, T₂ <: JoinTree{<:Any, T₁}}
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
    answer_query!(jt::JoinTree, query)

Answer a query, caching intermediate computations in `jt`.
"""
function answer_query!(jt::T₂, query) where {T₁, T₂ <: JoinTree{<:Any, T₁}}
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

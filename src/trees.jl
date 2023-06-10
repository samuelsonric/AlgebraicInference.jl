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
"""
function JoinTree{T₁, T₂}(kb, order) where {T₁, T₂ <: Valuation{T₁}}
    JoinTree{T₁, T₂}(map(ϕ -> convert(T₂, ϕ), kb), order) 
end

function JoinTree{T₁, T₂}(kb::Vector{<:T₂}, order) where {T₁, T₂ <: Valuation{T₁}}
    kb = copy(kb)
    pg = primalgraph(kb)
    ns = JoinTree{T₁, T₂}[]
    for i in 1:length(order)
        v = order[i]
        factor = one(T₂)
        for j in length(kb):-1:1
            if v in domain(kb[j])
                factor = combine(factor, kb[j])
                deleteat!(kb, j)
            end
        end
        node = JoinTree{T₁, T₂}(i, [v, neighbor_labels(pg, v)...], factor)
        for j in length(ns):-1:1
            if v in ns[j].domain
                ns[j].parent = node
                push!(node.children, ns[j])
                deleteat!(ns, j)
            end
        end
        push!(ns, node)
        eliminate!(pg, code_for(pg, v))
    end
    factor = one(T₂)
    for j in length(kb):-1:1
        factor = combine(factor, kb[j])
    end
    node = JoinTree{T₁, T₂}(length(order) + 1, collect(labels(pg)), factor)
    for j in length(ns):-1:1
        ns[j].parent = node
        push!(node.children, ns[j])
    end
    node
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
    solve(jt::JoinTree)
"""
function solve(jt::JoinTree)
    solve(jt, jt.domain)
end

"""
    solve!(jt::JoinTree)
"""
function solve!(jt::JoinTree)
    solve!(jt, jt.domain)
end

"""
    solve(jt::JoinTree, query)

Answer a query.
"""
function solve(jt::T₂, query) where {T₁, T₂ <: JoinTree{<:Any, T₁}}
    js = collect(Set(query))
    for node::T₂ in PreOrderDFS(jt)
        if js ⊆ node.domain        
            factor = node.factor
            for child in node.children
                factor = combine(factor, message_to_parent(child)::T₁)
            end
            if !isroot(node)
                factor = combine(factor, message_from_parent(node)::T₁)
            end
            return duplicate(project(factor, js), query)
        end 
    end
    error("Query not covered by join tree.")
end

"""
    solve!(jt::JoinTree, query)

Answer a query, caching intermediate computations in `jt`.
"""
function solve!(jt::T₂, query) where {T₁, T₂ <: JoinTree{<:Any, T₁}}
    js = collect(Set(query))
    for node::T₂ in PreOrderDFS(jt)
        if js ⊆ node.domain        
            factor = node.factor
            for child in node.children
                factor = combine(factor, message_to_parent!(child)::T₁)
            end
            if !isroot(node)
                factor = combine(factor, message_from_parent!(node)::T₁)
            end
            return duplicate(project(factor, js), query)
        end 
    end
    error("Query not covered by join tree.")
end

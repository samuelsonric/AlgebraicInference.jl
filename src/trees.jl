"""
    JoinTree{T₁ <: Valuation, T₂} <: AbstractNode{Int}

A join tree with factors of type `T₁` and variables of type `T₂`.
"""
mutable struct JoinTree{T₁ <: Valuation, T₂} <: AbstractNode{Int}
    id::Int
    factor::T₁
    domain::Vector{T₂}
    children::Vector{JoinTree{T₁, T₂}}
    parent::Union{Nothing, JoinTree{T₁, T₂}}
    message_from_parent::Union{Nothing, T₁}
    message_to_parent::Union{Nothing, T₁}

    function JoinTree{T₁, T₂}(id, factor, domain) where {T₁, T₂}
        new{T₁, T₂}(id, factor, domain, JoinTree{T₁, T₂}[], nothing, nothing, nothing)
    end
end

function JoinTree{T₁, T₂}(kb::Vector, order) where {T₁, T₂}
    kb = copy(kb)
    pg = primalgraph(kb)
    ns = JoinTree{T₁, T₂}[]
    for i in 1:length(order)
        v = order[i]
        factor = one(T₁)
        for j in length(kb):-1:1
            if v in domain(kb[j])
                factor = combine(factor, kb[j])
                deleteat!(kb, j)
            end
        end
        dom = [v, neighbor_labels(pg, v)...]
        node = JoinTree{T₁, T₂}(i, factor, dom)
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
    factor = one(T₁)
    for j in length(kb):-1:1
        factor = combine(factor, kb[j])
    end
    dom = collect(labels(pg))
    node = JoinTree{T₁, T₂}(length(order) + 1, factor, dom)
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
    solve(jt::JoinTree, query)

Answer a query.
"""
function solve(jt::T₂, query) where {T₁, T₂ <: JoinTree{T₁}}
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
function solve!(jt::T₂, query) where {T₁, T₂ <: JoinTree{T₁}}
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

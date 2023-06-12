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
    pg, ls = primalgraph(kb)
    ns = JoinTree{T₁, T₂}[]
    for i in 1:length(order)
        l = order[i]
        v = findfirst(_l -> _l == l, ls)
        factor = one(T₁)
        for j in length(kb):-1:1
            if l in domain(kb[j])
                factor = combine(factor, kb[j])
                deleteat!(kb, j)
            end
        end
        dom = [l; map(_v -> ls[_v], neighbors(pg, v))]
        node = JoinTree{T₁, T₂}(i, factor, dom)
        for j in length(ns):-1:1
            if l in ns[j].domain
                ns[j].parent = node
                push!(node.children, ns[j])
                deleteat!(ns, j)
            end
        end
        push!(ns, node)
        eliminate!(pg, ls, v)
    end
    factor = one(T₁)
    for j in length(kb):-1:1
        factor = combine(factor, kb[j])
    end
    dom = ls
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

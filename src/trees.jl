mutable struct JoinTree{T} <: AbstractNode{Int}
    factor::Valuation{T}
    id::Int
    domain::Vector{Int}
    children::Vector{JoinTree{T}}
    parent::Union{Nothing, JoinTree{T}}
    message_from_parent::Union{Nothing, Valuation{T}}
    message_to_parent::Union{Nothing, Valuation{T}}

    function JoinTree(factor::Valuation{T}, id, domain) where T
        new{T}(factor, id, domain, JoinTree{T}[], nothing, nothing, nothing)
    end
end

function JoinTree(kb::Vector{Valuation{T}}, pg::AbstractGraph, order) where T
    pg = copy(pg)
    ls = collect(vertices(pg))
    vs = collect(vertices(pg))
    ns = JoinTree{T}[]
    vpll = map(_ -> Set{Int}(), ls)
    for j in 1:length(kb)
        for js in vpll[domain(kb[j])]
            push!(js, j)
        end
    end
    for i in 1:length(order)
        l = order[i]
        v = vs[l]
        factor = one(Valuation{T})
        for j in vpll[l]
            factor = combine(factor, kb[j])
            for js in vpll[domain(kb[j])]
                delete!(js, j)
            end
        end
        node = JoinTree(factor, i, [l; ls[neighbors(pg, v)]])
        for j in length(ns):-1:1
            if l in ns[j].domain
                ns[j].parent = node
                push!(node.children, ns[j])
                deleteat!(ns, j)
            end
        end
        vs[ls[end]] = v
        push!(ns, node)
        eliminate!(pg, ls, v)
    end
    ns[end]
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

function nodetype(::Type{JoinTree{T}}) where T
    JoinTree{T}
end

function nodevalue(node::JoinTree)
    node.id
end

function parent(node::JoinTree)
    node.parent
end

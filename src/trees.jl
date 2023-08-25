mutable struct JoinTree{T} <: AbstractNode{Int}
    id::Int
    variables::Vector{Int}
    factors::Vector{Factor{T}}
    children::Vector{JoinTree{T}}
    parent::Union{Nothing, JoinTree{T}}
    message_from_parent::Union{Nothing, Factor{T}}
    message_to_parent::Union{Nothing, Factor{T}}

    function JoinTree{T}(id) where T
        new{T}(id, Int[], Factor{T}[], JoinTree{T}[], nothing, nothing, nothing)
    end
end

function JoinTree{T}(factors, graph, order) where T
    graph = copy(graph)
    nodes = JoinTree{T}[]
    vpll  = [Int[] for _ in graph.vertices]

    for j in eachindex(factors), js in vpll[factors[j].variables]
        push!(js, j)
    end

    for (i, v) in enumerate(order)
        node = JoinTree{T}(i)
        push!(node.variables, v)
        append!(node.variables, Graphs.neighbors(graph, v))

        for j in copy(vpll[v])
            push!(node.factors, factors[j])

            for js in vpll[factors[j].variables]
                deleteat!(js, js.==j)
            end
        end

        for j in reverse(eachindex(nodes))
            if v in nodes[j].variables
                nodes[j].parent = node
                push!(node.children, nodes[j])
                deleteat!(nodes, j)
            end
        end

        push!(nodes, node)
        eliminate!(graph, v)
    end

    nodes[end]
end

function AbstractTrees.ChildIndexing(::Type{<:JoinTree})
    IndexedChildren()
end

function AbstractTrees.NodeType(::Type{<:JoinTree})
    HasNodeType()
end

function AbstractTrees.ParentLinks(::Type{<:JoinTree})
    StoredParents()
end

function AbstractTrees.children(node::JoinTree)
    node.children
end

function AbstractTrees.nodetype(::Type{JoinTree{T}}) where T
    JoinTree{T}
end

function AbstractTrees.nodevalue(node::JoinTree)
    node.id
end

function AbstractTrees.parent(node::JoinTree)
    node.parent
end

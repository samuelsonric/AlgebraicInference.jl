mutable struct JoinTree{T} <: AbstractNode{Int}
    id::Int
    factors::Vector{Int}
    variables::Vector{T}
    parent::Union{Nothing, JoinTree{T}}
    children::Vector{JoinTree{T}}

    function JoinTree{T}(id, factors, variables) where T
        new(id, factors, variables, nothing, JoinTree{T}[])
    end
end

function JoinTree(
    f_to_v::Vector{Vector{T}},
    graph::LabeledGraph{T},
    order::Vector{T}) where T

    v_to_f = Dict(v => Int[] for v in Graphs.vertices(graph))

    for f in eachindex(f_to_v), v in f_to_v[f]
        push!(v_to_f[v], f)
    end

    graph = copy(graph)
    nodes = JoinTree{T}[]

    for (i, v) in enumerate(order)
        factors = copy(v_to_f[v])
        variables = [v; Graphs.neighbors(graph, v)]
        node = JoinTree{T}(i, factors, variables)
 
        for f in v_to_f[v], _v in f_to_v[f]
            if v != _v
                setdiff!(v_to_f[_v], f)
            end
        end

        for n in reverse(eachindex(nodes))
            if v in nodes[n].variables
                nodes[n].parent = node
                push!(node.children, nodes[n])    
                deleteat!(nodes, n)
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

function AbstractTrees.nodetype(::Type{JoinTree})
    JoinTree
end

function AbstractTrees.nodevalue(node::JoinTree)
    node.id
end

function AbstractTrees.parent(node::JoinTree)
    node.parent
end

"""
    EliminationAlgorithm

A graph elimination algorithm. The options are
- [`MinDegree`](@ref)
- [`MinFill`](@ref)
- [`MaxCardinality`](@ref)
- [`ChordalGraph`](@ref)
- [`CuthillMcKeeJL_RCM`](@ref)
- [`AMDJL_AMD`](@ref)
- [`MetisJL_ND`](@ref)
"""
abstract type EliminationAlgorithm end


"""
    MinDegree <: EliminationAlgorithm

The minimum-degree heuristic.
"""
struct MinDegree <: EliminationAlgorithm end


"""
    MinFill <: EliminationAlgorithm

The minimum-fill heuristic.
"""
struct MinFill <: EliminationAlgorithm end


"""
    MaxCardinality <: EliminationAlgorithm

The maximum cardinality search algorithm.
"""
struct MaxCardinality <: EliminationAlgorithm end


"""
    ChordalGraph <: EliminationAlgorithm

An efficient algorithm for chordal graphs.
"""
struct ChordalGraph <: EliminationAlgorithm end


"""
    CuthillMcKeeJL_RCM <: EliminationAlgorithm

The reverse Cuthill-McKee algorithm. Uses CuthillMckee.jl.
"""
struct CuthillMcKeeJL_RCM <: EliminationAlgorithm end


"""
    AMDJL_AMD <: EliminationAlgorithm

The approximate minimum degree algorithm. Uses AMD.jl.
"""
struct AMDJL_AMD  <: EliminationAlgorithm end


"""
    MetisJL_ND <: EliminationAlgorithm

The nested dissection heuristic. Uses Metis.jl.
"""
struct MetisJL_ND <: EliminationAlgorithm end


"""
    SupernodeType

A type of supernode. The options are
- [`Node`](@ref)
- [`MaximalSupernode`](@ref)
"""
abstract type SupernodeType end


"""
    Node <: SupernodeType

The single-vertex supernode.
"""
struct Node <: SupernodeType end


"""
    MaximalSupernode <: SupernodeType

The maximal supernode.
"""
struct MaximalSupernode <: SupernodeType end


"""
    FundamentalSupernode <: SupernodeType

The fundamental supernode.
"""
struct FundamentalSupernode <: SupernodeType end


# An ordering of the numbers {1, ..., n}.
struct Order <: AbstractVector{Int}
    order::Vector{Int}
    index::Vector{Int}
end


# An ordered graph
struct OrderedGraph <: Graphs.AbstractGraph{Int}
    order::Order
    graph::Graphs.Graph{Int}
end


# An elimination tree.
struct EliminationTree <: AbstractVector{Vector{Int}}
    order::Order
    parent::Vector{Int}               # pa(v)
    children::Vector{Vector{Int}}     # ch(v)
    outneighbors::Vector{Vector{Int}} # adj⁺(v) (sorted)
end


# A join tree.
struct JoinTree <: AbstractVector{Tuple{Vector{Int}, Vector{Int}}}
    order::Order
    parent::Vector{Int}             # pa(v)
    children::Vector{Vector{Int}}   # ch(v)
    seperators::Vector{Vector{Int}} # sep(v) (sorted)
    residuals::Vector{Vector{Int}}  # res(v) (sorted)
end


# Determine if
# v₁ < v₂
# in the given order.
function (order::Order)(v₁::Integer, v₂::Integer)
    order.index[v₁] < order.index[v₂]
end


function Order(order::AbstractVector)
    n = length(order)
    index = Vector{Int}(undef, n)
    index[order] .= 1:n

    Order(order, index)
end


# Construct an order of length n.
function Order(n::Integer)
    order = Vector{Int}(undef, n)
    index = Vector{Int}(undef, n)

    Order(order, index)
end


# Construct an elimination order using the minimum-degree heuristic.
function Order(graph::Graphs.AbstractGraph, elimination_algorithm::MinDegree)
    mindegree!(copy(graph))
end


# Construct an elimination order using the minimum-fill heuristic.
function Order(graph::Graphs.AbstractGraph, elimination_algorithm::MinFill)
    minfill!(copy(graph))
end


# Construct an elimination order using the maximum cardinality search algorithm.
function Order(graph::Graphs.AbstractGraph, elimination_algorithm::MaxCardinality)
    maxcardinality(graph)
end


# Construct a perfect elimination order for a chordal graph.
function Order(graph::Graphs.AbstractGraph, elimination_algorithm::ChordalGraph)
    Order(graph, MaxCardinality())
end


# Construct an elimination order using the reverse Cuthill-McKee algorithm. Uses
# CuthillMcKee.jl.
function Order(graph::Graphs.AbstractGraph, elimination_algorithm::CuthillMcKeeJL_RCM)
    order = CuthillMcKee.symrcm(Graphs.adjacency_matrix(graph))
    Order(order)
end


# Construct an elimination order using the approximate minimum degree algorithm. Uses
# AMD.jl.
function Order(graph::Graphs.AbstractGraph, elimination_algorithm::AMDJL_AMD)
    order = AMD.symamd(Graphs.adjacency_matrix(graph))
    Order(order)
end


# Construct an elimination order using the nested dissection heuristic. Uses Metis.jl.
function Order(graph::Graphs.AbstractGraph, elimination_algorithm::MetisJL_ND)
    order, index = Metis.permutation(graph)
    Order(order, index)
end


# Construct an elimination tree using the given elimination algorithm.
function EliminationTree(
    graph::Graphs.AbstractGraph,
    elimination_algorithm::EliminationAlgorithm)

    order = Order(graph, elimination_algorithm)
    ordered_graph = OrderedGraph(order, graph)
    EliminationTree(ordered_graph, Val(false))
end


# Construct an elimination tree for a chordal graph.
function EliminationTree(
    graph::Graphs.AbstractGraph,
    elimination_algorithm::ChordalGraph)

    order = Order(graph, elimination_algorithm)
    ordered_graph = OrderedGraph(order, graph)
    EliminationTree(ordered_graph, Val(true))
end


# Construct the elimination tree of the elimination graph of an ordered graph.
# Algorithm 4.2 in doi:10.1145/6497.6499.
function EliminationTree(graph::OrderedGraph, isfilled::Val{false})
    n = Graphs.nv(graph)
    order = graph.order

    ancestor = Vector{Int}(undef, n)
    parent = Vector{Int}(undef, n)
    children = Vector{Vector{Int}}(undef, n)
    outneighbors = Vector{Vector{Int}}(undef, n)

    for v in order
        ancestor[v] = 0
        parent[v] = 0
        children[v] = Int[] 
        outneighbors[v] = Graphs.outneighbors(graph, v) 
 
        for w in Graphs.inneighbors(graph, v)
            u = w
            
            while ancestor[u] != 0 && ancestor[u] != v
                t = ancestor[u]
                ancestor[u] = v
                u = t
            end
            
            if ancestor[u] == 0
                ancestor[u] = v
                parent[u] = v
                push!(children[v], u)

                for t in outneighbors[u]
                    if t != v
                        insertsorted!(outneighbors[v], t)
                    end
                end
            end
        end
    end

    # In case the graph is not connected.
    for v in order[1:end - 1]
        if parent[v] == 0
            parent[v] = order[end]
            push!(children[order[end]], v)
        end
    end

    EliminationTree(order, parent, children, outneighbors)
end


# Construct the elimination tree of a filled graph.
function EliminationTree(graph::OrderedGraph, isfilled::Val{true})
    n = Graphs.nv(graph)
    order = graph.order

    parent = Vector{Int}(undef, n)
    children = Vector{Vector{Int}}(undef, n)
    outneighbors = Vector{Vector{Int}}(undef, n)

    v = order[end]
    parent[v] = 0
    children[v] = Int[]
    outneighbors[v] = Int[]

    for v in order[end - 1:-1:1]
        ns = Graphs.outneighbors(graph, v)
        parent[v] = argmin(w -> order.index[w], ns)
        children[v] = Int[]
        outneighbors[v] = ns
        push!(children[parent[v]], v)
    end

    EliminationTree(order, parent, children, outneighbors)
end


# Construct a join tree, using a descent-first search to find a topological ordering of its
# nodes.
function JoinTree(
    rootindex::Int,
    parent::Vector{Int},
    children::Vector{Vector{Int}},
    seperators::Vector{Vector{Int}},
    residuals::Vector{Vector{Int}})

    
    order = Order(length(parent))
    order[end] = rootindex

    tree = JoinTree(order, parent, children, seperators, residuals)

    for (i, node) in enumerate(PostOrderDFS(IndexNode(tree)))
        order[i] = node.index
    end

    tree 
end


function JoinTree(
    graph::Graphs.AbstractGraph,
    elimination_algorithm::EliminationAlgorithm,
    supernode_type::SupernodeType)

    elimination_tree = EliminationTree(graph, elimination_algorithm)
    JoinTree(elimination_tree, supernode_type)
end


# Construct a nodal elimination tree.
function JoinTree(tree::EliminationTree, supernode_type::Node)
    order = tree.order
    parent = tree.parent
    children = tree.children
    seperators = tree.outneighbors
    residuals = [[i] for i in eachindex(tree)]
    
    JoinTree(order, parent, children, seperators, residuals)
end


# Construct a supernodal elimination tree.
# Algorithm 4.1 in doi:10.1561/2400000006.
function JoinTree(tree::EliminationTree, supernode_type::SupernodeType)
    parent = Vector{Int}()
    children = Vector{Vector{Int}}()
    residuals = Vector{Vector{Int}}()
    representatives = Vector{Int}()

    v_to_n = Vector{Int}(undef, length(tree))

    for v in tree.order
        ŵ = sndchild(tree, supernode_type, v)

        if ŵ == 0
            push!(parent, 0)
            push!(children, Int[])
            push!(residuals, Int[])
            push!(representatives, v)        

            n̂ = length(parent)
        else
            n̂ = v_to_n[ŵ]
        end

        insertsorted!(residuals[n̂], v)
        v_to_n[v] = n̂

        for w in childindices(tree, v)
            n = v_to_n[w]

            if ŵ != w
                parent[n] = n̂
                push!(children[n̂], n) 
            end 
        end 
    end

    m = length(representatives)
    seperators = Vector{Vector{Int}}(undef, m)

    for i in 1:m
        v = representatives[i]
        seperators[i] = filter(w -> !insorted(w, residuals[i]), tree[v])
    end

    rootindex = v_to_n[tree.order[end]]
    JoinTree(rootindex, parent, children, seperators, residuals)
end


# Get the width of an elimination tree.
function width(tree::EliminationTree)
    maximum(length, tree)
end


# Get the width of a join tree.
function width(tree::JoinTree)
    maximum(vars -> sum(map(length, vars)) - 1, tree)
end


#=
function sndchild(tree::EliminationTree, supernode_type::Node, v::Integer)
    0
end
=#


# If v is the representative vertex of a supernode, returns 0.
# Otherwise, returns a child of v whose supernode contains v.
function sndchild(tree::EliminationTree, supernode_type::MaximalSupernode, v::Integer)
    for w in childindices(tree, v)
        if length(tree[w]) == length(tree[v]) + 1
            return w
        end
    end

    0
end


function sndchild(tree::EliminationTree, supernode_type::FundamentalSupernode, v::Integer)
    if length(childindices(tree, v)) == 1
        w = only(childindices(tree, v))

        if length(tree[w]) == length(tree[v]) + 1
            return w
        end
    end 

    0
end


# Get the fill-in number of vertex v.
function fillin(graph::Graphs.AbstractGraph, v::Integer)
    count = 0
    ns = Graphs.neighbors(graph, v)
    n = length(ns)

    for i₁ in 1:n - 1, i₂ in i₁ + 1:n
        if !Graphs.has_edge(graph, ns[i₁], ns[i₂])
            count += 1
        end
    end

    count
end


# Compute an elimination order using the minimum-degree heuristic.
function mindegree!(graph::Graphs.AbstractGraph)
    n = Graphs.nv(graph)
    order = Order(n)
    labels = Labels(1:n)

    for i in 1:n
        v = _argmin(v -> Graphs.degree(graph, v), Graphs.vertices(graph), 1)
        l = labels[v]
        order[i] = l
        eliminate!(labels, graph, l)
    end
    
    order
end


# Compute a vertex elimination order using the minimum-fill heuristic.
function minfill!(graph::Graphs.AbstractGraph)
    n = Graphs.nv(graph)
    order = Order(n)
    labels = Labels(1:n)
    fillins = [fillin(graph, v) for v in 1:n]
    
    for i in 1:n
        v = _argmin(fillins, 0)
        l = labels[v]
        order[i] = l
        eliminate!(labels, graph, fillins, l)
    end

    order
end


# Compute a vertex elimination order using the maximum cardinality search algorithm.
# Page 569 of doi:10.1007/s10878-018-0270-1.
function maxcardinality(graph::Graphs.AbstractGraph)
    n = Graphs.nv(graph)
    order = Order(n)

    size = Vector{Int}(undef, n)
    set = Vector{Vector{Int}}(undef, n)

    size[1] = 1
    set[1] = 1:n

    for v in 2:n
        size[v] = 1
        set[v] = Int[]
    end

    j = 1

    for i in n:-1:1
        v = pop!(set[j])
        order[i] = v
        size[v] = 0

        for w in Graphs.neighbors(graph, v)
            if size[w] >= 1
                deletesorted!(set[size[w]], w)
                
                size[w] += 1
                insertsorted!(set[size[w]], w)
            end
        end

        j += 1

        while j >= 1 && isempty(set[j])
            j -= 1
        end
    end

    order
end


# Eliminate the vertex v.
function eliminate!(labels::Labels, graph::Graphs.AbstractGraph, l)
    v = labels.index[l]
    ns = Graphs.neighbors(graph, v)
    n = length(ns)

    for i₁ in 1:n - 1, i₂ in i₁ + 1:n
        Graphs.add_edge!(graph, ns[i₁], ns[i₂])
    end

    delete!(labels, l)
    Graphs.rem_vertex!(graph, v)
end


# Eliminate the vertex v.
# Adapted from https://github.com/JuliaQX/QXGraphDecompositions.jl/blob/
# 22ee3d75bcd267bf462eec8f03930af2129e34b7/src/LabeledGraph.jl#L326
function eliminate!(labels::Labels, graph::Graphs.AbstractGraph, fillins::Vector{Int}, l)
    v = labels.index[l]
    ns = Graphs.neighbors(graph, v)
    n = length(ns)

    for i₁ = 1:n - 1, i₂ = i₁ + 1:n
        u₁ = ns[i₁]
        u₂ = ns[i₂]

        if Graphs.add_edge!(graph, u₁, u₂)
            ns₁ = Graphs.neighbors(graph, u₁)
            ns₂ = Graphs.neighbors(graph, u₂)

            for w in ns₁ ∩ ns₂
                fillins[w] -= 1
            end

            for w in ns₁
                if w != u₂ && !Graphs.has_edge(graph, w, u₂)
                    fillins[u₁] += 1
                end
            end

            for w in ns₂
                if w != u₁ && !Graphs.has_edge(graph, w, u₁)
                    fillins[u₂] += 1
                end
            end
        end
    end

    for i in 1:n
        u = ns[i]
        nsᵤ = Graphs.neighbors(graph, u)

        for w in nsᵤ
            if w != v && !Graphs.has_edge(graph, w, v)
                fillins[u] -= 1
            end
        end
    end

    delete!(labels, l)
    Graphs.rem_vertex!(graph, v)

    fillins[v] = fillins[end]
    pop!(fillins)
end


############################
# AbstractVector Interface #
############################


function Base.size(A::Order)
    (length(A.order),)
end


function Base.size(A::EliminationTree)
    (length(A.parent),)
end


function Base.size(A::JoinTree)
    (length(A.parent),)
end


function Base.getindex(A::Order, n::Integer)
    A.order[n]
end


function Base.getindex(A::EliminationTree, n::Integer)
    A.outneighbors[n]
end


function Base.getindex(A::JoinTree, n::Integer)
    A.seperators[n], A.residuals[n]
end


function Base.IndexStyle(::Type{Order})
    IndexLinear()
end


function Base.IndexStyle(::Type{EliminationTree})
    IndexLinear()
end


function Base.IndexStyle(::Type{JoinTree})
    IndexLinear()
end


function Base.setindex!(A::Order, v::Integer, i::Integer)
    A.order[i] = v
    A.index[v] = i
end


##########################
# Indexed Tree Interface #
##########################


function AbstractTrees.rootindex(tree::EliminationTree)
    tree.order[end]
end


function AbstractTrees.rootindex(tree::JoinTree)
    tree.order[end]
end


function AbstractTrees.parentindex(tree::EliminationTree, n::Integer)
    n == rootindex(tree) ? nothing : tree.parent[n]
end


function AbstractTrees.parentindex(tree::JoinTree, n::Integer)
    n == rootindex(tree) ? nothing : tree.parent[n]
end


function AbstractTrees.childindices(tree::EliminationTree, n::Integer)
    tree.children[n]
end


function AbstractTrees.childindices(tree::JoinTree, n::Integer)
    tree.children[n]
end


function AbstractTrees.NodeType(::Type{IndexNode{EliminationTree, Int}})
    HasNodeType()
end


function AbstractTrees.NodeType(::Type{IndexNode{JoinTree, Int}})
    HasNodeType()
end


function AbstractTrees.nodetype(::Type{IndexNode{EliminationTree, Int}})
    IndexNode{EliminationTree, Int}
end


function AbstractTrees.nodetype(::Type{IndexNode{JoinTree, Int}})
    IndexNode{JoinTree, Int}
end


###########################
# AbstractGraph interface #
###########################


function Graphs.vertices(g::OrderedGraph)
    Graphs.vertices(g.graph)
end


function Graphs.nv(g::OrderedGraph)
    Graphs.nv(g.graph)
end


function Graphs.outneighbors(g::OrderedGraph, v::Integer)
    filter(u -> g.order(v, u), Graphs.neighbors(g.graph, v))
end


function Graphs.inneighbors(g::OrderedGraph, v::Integer)
    filter(u -> g.order(u, v), Graphs.neighbors(g.graph, v))
end

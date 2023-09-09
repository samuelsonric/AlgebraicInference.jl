"""
    EliminationAlgorithm

An algorithm for computing an elimination order for an undirected graph.
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


# An elimination order.
struct EliminationOrder <: AbstractVector{Int}
    order::Vector{Int}
    index::Vector{Int}
end


# An elimination tree.
struct EliminationTree
    rootindex::Int
    parent::Vector{Int}            # pa(v)
    children::Vector{Vector{Int}}  # ch(v)
    neighbors::Vector{Vector{Int}} # adj⁺(v)
end

# Return `true` if
# v₁ < v₂
# in the given order.
function (order::EliminationOrder)(v₁::Int, v₂::Int)
    order.index[v₁] < order.index[v₂]
end


function EliminationOrder(order::AbstractVector)
    n = length(order)
    index = Vector{Int}(undef, n)
    index[order] .= 1:n

    EliminationOrder(order, index)
end


function EliminationOrder(n::Integer)
    order = Vector{Int}(undef, n)
    index = Vector{Int}(undef, n)

    EliminationOrder(order, index)
end


# Construct an elimination order using the minimum-degree heuristic.
function EliminationOrder(graph::Graphs.Graph, alg::MinDegree)
    mindegree!(copy(graph))
end


# Construct an elimination order using the minimum-fill heuristic.
function EliminationOrder(graph::Graphs.Graph, alg::MinFill)
    minfill!(copy(graph))
end


# Construct an elimination order using the reverse Cuthill-McKee algorithm. Uses
# CuthillMcKee.jl.
function EliminationOrder(graph::Graphs.Graph, alg::CuthillMcKeeJL_RCM)
    order = CuthillMcKee.symrcm(Graphs.adjacency_matrix(graph))
    EliminationOrder(order)
end


# Construct an elimination order using the approximate minimum degree algorithm. Uses
# AMD.jl.
function EliminationOrder(graph::Graphs.Graph, alg::AMDJL_AMD)
    order = AMD.symamd(Graphs.adjacency_matrix(graph))
    EliminationOrder(order)
end


# Construct an elimination order using the nested dissection heuristic. Uses Metis.jl.
function EliminationOrder(graph::Graphs.Graph, alg::MetisJL_ND)
    order, index = Metis.permutation(graph)
    EliminationOrder(order, index)
end


# Construct the elimination tree of the elimination graph determined by the given ordered
# graph. 
function EliminationTree(graph::Graphs.Graph, order::EliminationOrder)
    parent = path_compression(graph, order)

    children = [Int[] for _ in order]
    neighbors = [Int[] for _ in order]

    for v in order[1:end - 1]
        u = parent[v]
        union!(neighbors[v], adj⁺(graph, order, v))
        push!(children[u], v)

        for t in neighbors[v]
            if t != v && t != u
                push!(neighbors[u], t)
            end
        end
    end
   
    v = order[end] 
    union!(neighbors[v], adj⁺(graph, order, v))

    EliminationTree(v, parent, children, neighbors)
end


# Compute the higher neighborhood
# adj⁺(v)
# of v.
function adj⁺(graph::Graphs.Graph, order::EliminationOrder, v::Int)
    ns = Graphs.neighbors(graph, v)
    filter(u -> order(v, u), ns)
end


# Compute the lower neighborhood
# adj⁻(v)
# of v.
function adj⁻(graph::Graphs.Graph, order::EliminationOrder, v::Int)
    ns = Graphs.neighbors(graph, v)
    filter(u -> order(u, v), ns)
end


# Compute the parent function of the elimination tree of the elimination graph determined
# by the given ordered graph.
#
# Uses Algorithm 4.2 in https://dl.acm.org/doi/pdf/10.1145/6497.6499
function path_compression(graph::Graphs.Graph, order::EliminationOrder)
    n = Graphs.nv(graph)
    parents = Vector{Int}(undef, n)
    ancestors = Vector{Int}(undef, n)
    
    for v in order
        parents[v] = 0
        ancestors[v] = 0
        
        for w in adj⁻(graph, order, v)
            u = w
            
            while ancestors[u] != 0 && ancestors[u] != v
                t = ancestors[u]
                ancestors[u] = v
                u = t
            end
            
            if ancestors[u] == 0
                ancestors[u] = v
                parents[u] = v
            end
        end
    end
    
    parents
end


# The fill-in number of vertex v.
function fillin(graph::Graphs.Graph, v::Int)
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


# Compute an elimination order using the minimum degree heuristic.
function mindegree!(graph::Graphs.Graph)
    n = Graphs.nv(graph)
    order = EliminationOrder(n)
    labels = Labels(1:n)

    for i in 1:n
        v = ssargmin(v -> Graphs.degree(graph, v), Graphs.vertices(graph), 1)
        l = labels[v]
        order[i] = l
        eliminate!(labels, graph, l)
    end
    
    order
end


# Compute a vertex elimination order using the minimum fill heuristic.
function minfill!(graph::Graphs.Graph)
    n = Graphs.nv(graph)
    order = EliminationOrder(n)
    labels = Labels(1:n)
    fillins = [fillin(graph, v) for v in 1:n]
    

    for i in 1:n
        v = ssargmin(fillins, 0)
        l = labels[v]
        order[i] = l
        eliminate!(labels, graph, fillins, l)
    end

    order
end



# Eliminate the vertex v.
function eliminate!(labels::Labels, graph::Graphs.Graph, l)
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
function eliminate!(labels::Labels, graph::Graphs.Graph, fillins::Vector{Int}, l)
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


function Base.size(A::EliminationOrder)
    (length(A.order),)
end


function Base.size(A::EliminationTree)
    (length(A.parent),)
end


function Base.getindex(A::EliminationOrder, i::Int)
    A.order[i]
end


function Base.getindex(A::EliminationTree, i::Int)
    A.neighbors[i]
end


function Base.IndexStyle(::Type{EliminationOrder})
    IndexLinear()
end


function Base.IndexStyle(::Type{EliminationTree})
    IndexLinear()
end


function Base.setindex!(A::EliminationOrder, v, i::Int)
    A.order[i] = v
    A.index[v] = i
end


##########################
# Indexed Tree Interface #
##########################


function AbstractTrees.rootindex(tree::EliminationTree)
    tree.rootindex
end


function AbstractTrees.parentindex(tree::EliminationTree, i::Int)
    i == rootindex(tree) ? nothing : tree.parent[i]
end


function AbstractTrees.childindices(tree::EliminationTree, i::Int)
    tree.children[i]
end

struct LabeledGraph <: Graphs.AbstractGraph{Int}
    graph::Graphs.Graph{Int}
    labels::Vector{Int}
    vertices::Vector{Int}

    function LabeledGraph(graph, labels, vertices)
        @assert Graphs.nv(graph) == length(labels)
        new(graph, labels, vertices)
    end
end

function LabeledGraph(n::Int)
    LabeledGraph(Graphs.Graph{Int}(n), 1:n, 1:n) 
end

function Graphs.add_edge!(g::LabeledGraph, v₁::Int, v₂::Int)
    Graphs.add_edge!(g.graph, g.vertices[v₁], g.vertices[v₂])
end

function Base.copy(g::LabeledGraph)
    LabeledGraph(copy(g.graph), copy(g.labels), copy(g.vertices))
end

function Graphs.has_edge(g::LabeledGraph, v₁::Int, v₂::Int)
    Graphs.has_edge(g.graph, g.vertices[v₁], g.vertices[v₂])
end

function Graphs.has_vertex(g::LabeledGraph, v::Int)
    Graphs.has_vertex(g.graph, g.vertices[v])
end

function Graphs.inneighbors(g::LabeledGraph, v::Int)
    ns = Graphs.inneighbors(g.graph, g.vertices[v])
    g.labels[ns]
end

function Graphs.is_directed(::Type{LabeledGraph})
    Graphs.is_directed(Graphs.Graph{Int})
end

function Graphs.ne(g::LabeledGraph)
    Graphs.ne(g.graph)
end

function Graphs.nv(g::LabeledGraph)
    Graphs.nv(g.graph)
end

function Graphs.degree(g::LabeledGraph, v::Int)
    Graphs.degree(g.graph, g.vertices[v])
end

function Graphs.outneighbors(g::LabeledGraph, v::Int)
    ns = Graphs.outneighbors(g.graph, g.vertices[v])
    g.labels[ns]
end

function Graphs.rem_vertex!(g::LabeledGraph, v::Int)
    w = g.vertices[v]
    Graphs.rem_vertex!(g.graph, w)
    g.labels[w] = g.labels[end]
    g.vertices[g.labels[w]] = w
    pop!(g.labels)
end

function Graphs.rem_edge!(g::LabeledGraph, v₁::Int, v₂::Int)
    Graphs.rem_edge!(g.graph, g.vertices[v₁], g.vertices[v₂])
end

function Graphs.vertices(g::LabeledGraph)
    g.labels
end

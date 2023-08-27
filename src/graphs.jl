struct LabeledGraph{T}
    variables::Vector{T}
    vertices::Dict{T, Int}
    graph::Graphs.Graph{Int}
 
   function LabeledGraph{T}(variables, vertices, graph) where T
        @assert length(variables) == length(vertices) == Graphs.nv(graph)
        new{T}(variables, vertices, graph)
    end
end

function LabeledGraph(
    variables::Vector{T},
    vertices::Dict{T, Int},
    graph::Graphs.Graph{Int}) where T

    LabeledGraph{T}(variables, vertices, graph)
end

function LabeledGraph{T}(variables) where T
    variables = Vector{T}(variables)
    vertices = Dict{T, Int}(map(reverse, enumerate(variables)))
    graph = Graphs.Graph{Int}(length(variables))
    LabeledGraph(variables, vertices, graph)
end

function Graphs.add_edge!(g::LabeledGraph, v₁, v₂)
    Graphs.add_edge!(g.graph, g.vertices[v₁], g.vertices[v₂])
end

function Base.copy(g::LabeledGraph)
    LabeledGraph(copy(g.variables), copy(g.vertices), copy(g.graph))
end

function Graphs.has_edge(g::LabeledGraph, v₁, v₂)
    Graphs.has_edge(g.graph, g.vertices[v₁], g.vertices[v₂])
end

function Graphs.has_vertex(g::LabeledGraph, v)
    Graphs.has_vertex(g.graph, g.vertices[v])
end

function Graphs.ne(g::LabeledGraph)
    Graphs.ne(g.graph)
end

function Graphs.neighbors(g::LabeledGraph, v)
    ns = Graphs.neighbors(g.graph, g.vertices[v])
    g.variables[ns]
end

function Graphs.nv(g::LabeledGraph)
    Graphs.nv(g.graph)
end

function Graphs.degree(g::LabeledGraph, v::Integer)
    Graphs.degree(g.graph, g.vertices[v])
end

function Graphs.degree(g::LabeledGraph, v)
    Graphs.degree(g.graph, g.vertices[v])
end

function Graphs.rem_vertex!(g::LabeledGraph, v)
    w = g.vertices[v]
    Graphs.rem_vertex!(g.graph, w)

    g.variables[w] = g.variables[end]
    g.vertices[g.variables[w]] = w

    pop!(g.variables)
    delete!(g.vertices, v)
end

function Graphs.vertices(g::LabeledGraph)
    g.variables
end

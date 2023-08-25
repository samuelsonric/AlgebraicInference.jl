"""
    GraphicalModel{T₁, T₂}

An undirected graphical model.
"""
mutable struct GraphicalModel{T₁, T₂}
    factors::Vector{Factor{T₁}}
    objects::Vector{T₂}
    graph::LabeledGraph
end

function GraphicalModel{T₁, T₂}(m::Integer, n::Integer) where {T₁, T₂}
    factors = Vector{Factor{T₁}}(undef, m)
    objects = Vector{T₂}(undef, n)
    graph = LabeledGraph(n)
    GraphicalModel{T₁, T₂}(factors, objects, graph)
end

function Base.copy(gm::GraphicalModel)
    GraphicalModel(copy(gm.factors), copy(gm.objects), copy(gm.graph))
end

"""
    context!(gm::GraphicalModel, evidence)

Reduce the graphical model `gm` to to the context specified by `evidence`.
"""
function context!(gm::GraphicalModel, evidence::AbstractDict{Int})
    for (i, fac) in enumerate(gm.factors), v in fac.variables
        if haskey(evidence, v)
            gm.factors[i] = context(gm.factors[i], evidence[v], v, gm.objects)
        end
    end

    for v in keys(evidence)
        Graphs.rem_vertex!(gm.graph, v)
    end
end

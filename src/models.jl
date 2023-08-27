"""
    GraphicalModel{T₁, T₂, T₃}

An undirected graphical model.
"""
mutable struct GraphicalModel{T₁, T₂, T₃}
    factors::Vector{Factor{T₁, T₂}}
    objects::Dict{T₁, T₃}
    graph::LabeledGraph{T₁}
end

mutable struct JoinTreeModel{T₁, T₂, T₃}
    factors::Vector{Factor{T₁, T₂}}
    objects::Dict{T₁, T₃}
    tree::JoinTree{T₁}
    mailboxes::Matrix{Union{Nothing, Factor{T₁, T₂}}}
end

"""
    GraphicalModel{T₁, T₂, T₃}(
        fg::AbstractUndirectedBipartiteGraph,
        homs::AbstractVector,
        obs::AbstractVector) where {T₁, T₂, T₃}
"""
function GraphicalModel{T₁, T₂, T₃}(
    fg::AbstractUndirectedBipartiteGraph,
    homs::AbstractVector,
    obs::AbstractVector) where {T₁, T₂, T₃}

    @assert nv₁(fg) == length(homs)
    @assert nv₂(fg) == length(obs)

    factors = Vector{Factor{T₁, T₂}}(undef, nv₁(fg))
    objects = Dict{T₁, T₃}(enumerate(obs))
    graph = LabeledGraph{T₁}(vertices₂(fg))

    i = 1

    for i₁ in edges(fg)
        for i₂ in i:i₁ - 1
            if tgt(fg, i₁) != tgt(fg, i₂)
                Graphs.add_edge!(graph, tgt(fg, i₁), tgt(fg, i₂))
            end
        end

        if src(fg, i) != src(fg, i₁)
            factors[src(fg, i)] = Factor(tgt(fg, i:i₁ - 1), homs[src(fg, i)]) 
            i = i₁
        end
    end

    factors[end] = Factor(tgt(fg, i:ne(fg)), homs[end])

    GraphicalModel(factors, objects, graph)
end

"""
    GraphicalModel{T₁, T₂, T₃}(bn::BayesNet) where {T₁, T₂, T₃}
"""
function GraphicalModel{T₁, T₂, T₃}(bn::BayesNet) where {T₁, T₂, T₃}
    n = length(bn)

    factors = Vector{Factor{T₁, T₂}}(undef, n)
    objects = Dict{T₁, T₃}(enumerate(ones(T₃, n)))
    graph = LabeledGraph{T₁}(1:n)

    for i in 1:n
        cpd = bn.cpds[i]
        pas = [bn.name_to_index[l] for l in parents(cpd)] 
 
        factors[i] = Factor([pas; i], cpd)

        for j₁ in eachindex(pas)
            Graphs.add_edge!(graph, pas[j₁], i)

            for j₂ in 1:j₁ - 1
                Graphs.add_edge!(graph, pas[j₁], pas[j₂])
            end
        end
    end

    GraphicalModel(factors, objects, graph)
end

function JoinTreeModel(
    factors::Vector{Factor{T₁, T₂}},
    objects::Dict{T₁},
    tree::JoinTree{T₁}) where {T₁, T₂}

    mailboxes = Matrix{Union{Nothing, Factor{T₁, T₂}}}(nothing, 2, length(factors))
    JoinTreeModel(factors, objects, tree, mailboxes)
end

function JoinTreeModel(model::GraphicalModel, order::AbstractVector)
    f_to_v = [fac.variables for fac in model.factors]
    tree = JoinTree(f_to_v, model.graph, order)
    JoinTreeModel(model.factors, model.objects, tree)
end

function Base.copy(model::GraphicalModel)
    GraphicalModel(copy(model.factors), copy(model.objects), copy(model.graph))
end

"""
    observe!(model::GraphicalModel, evidence::AbstractDict)

Reduce the graphical model `model` to to the observe specified by `evidence`.
"""
function observe!(model::GraphicalModel, evidence::AbstractDict)
    for (i, fac) in enumerate(model.factors), v in fac.variables
        if haskey(evidence, v)
            model.factors[i] = observe(model.factors[i], evidence[v], v, model.objects)
        end
    end

    for v in keys(evidence)
        Graphs.rem_vertex!(model.graph, v)
        delete!(model.objects, v)
    end
end

function observe!(model::JoinTreeModel, evidence::AbstractDict)
     for (i, fac) in enumerate(model.factors), v in fac.variables
        if haskey(evidence, v)
            model.factors[i] = observe(model.factors[i], evidence[v], v, model.objects)
        end
    end

    for v in keys(evidence)
        delete!(model.objects, v)
    end

    for node in PreOrderDFS(model.tree)
        setdiff!(node.variables, keys(evidence))
    end
end

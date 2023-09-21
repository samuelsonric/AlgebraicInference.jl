# An undirected graphical model.
mutable struct GraphicalModel{T₁, T₂, T₃}
    labels::Labels{T₁}
    factors::Vector{Factor{T₂, T₃}}
    graph::Graphs.Graph{Int}
    vvll::Vector{Vector{Int}}
end


function GraphicalModel{T₁, T₂, T₃}(
    factor_graph::AbstractUndirectedBipartiteGraph,
    labels::AbstractVector,
    morphisms::AbstractVector,
    objects::AbstractVector) where {T₁, T₂, T₃}

    @assert nv₁(factor_graph) == length(morphisms)
    @assert nv₂(factor_graph) == length(labels) == length(objects)

    scopes = [Int[] for _ in vertices₁(factor_graph)]
    vvll = [Int[] for _ in vertices₂(factor_graph)]

    for i in edges(factor_graph)
        f = src(factor_graph, i)
        v = tgt(factor_graph, i)

        push!(scopes[f], v)
        push!(vvll[v], f)
    end

    factors = Vector{Factor{T₂, T₃}}(undef, nv₁(factor_graph))
    graph = Graphs.Graph(nv₂(factor_graph))

    for (f, vs) in enumerate(scopes)
        n = length(vs)

        for i₁ in 2:n, i₂ in 1:i₁ - 1
            Graphs.add_edge!(graph, vs[i₁], vs[i₂])
        end

        factors[f] = Factor(morphisms[f], objects[vs], vs)
    end

    labels = Labels{T₁}(labels)
    GraphicalModel(labels, factors, graph, vvll)
end


function GraphicalModel{T₁, T₂, T₃}(network::BayesNets.BayesNet) where {T₁, T₂, T₃}
    n = length(network)

    labels = Labels{T₁}(names(network))
    factors = Vector{Factor{T₂, T₃}}(undef, n)
    graph = Graphs.Graph{Int}(n)
    vvll = [[i] for i in 1:n]

    for i in 1:n
        cpd = network.cpds[i]
        parents =  map(l -> network.name_to_index[l], BayesNets.parents(cpd))
        m = length(parents)
  
        morphism = GaussianSystem(cpd)
        objects = ones(Int, m + 1)
        variables = [parents; i]

        factors[i] = Factor(morphism, objects, variables)

        for j₁ in 1:m
            i₁ = parents[j₁]
            push!(vvll[i₁], i)
            Graphs.add_edge!(graph, i₁, i)

            for j₂ in 1:j₁ - 1
                i₂ = parents[j₂]
                Graphs.add_edge!(graph, i₁, i₂)
            end
        end
    end

    GraphicalModel(labels, factors, graph, vvll)
end


function reduce_to_context(model::GraphicalModel, context::AbstractDict)
    labels = copy(model.labels)
    factors = copy(model.factors)
    graph = copy(model.graph)
    vvll = copy(model.vvll)

    for (l, hom) in context
        v = labels.index[l]

        for f in vvll[v]
            factors[f] = reduce_to_context(factors[f], v => hom)
        end

        for f in vvll[end]
            fac = factors[f]
            factors[f] = Factor(fac.hom, fac.obs, replace(fac.vars, length(vvll) => v))
        end

        delete!(labels, l)
        Graphs.rem_vertex!(graph, v)
    
        vvll[v] = vvll[end]
        pop!(vvll)
    end

    GraphicalModel(labels, factors, graph, vvll)
end

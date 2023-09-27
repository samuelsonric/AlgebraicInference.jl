# An undirected graphical model.
mutable struct GraphicalModel{T₁, T₂, T₃}
    labels::Labels{T₁}
    factors::Vector{Factor{false, T₂, T₃}}
    graph::Graphs.Graph{Int}
    vvll::Vector{Vector{Int}}
end


function GraphicalModel{T₁, T₂, T₃}(
    factorgraph::AbstractUndirectedBipartiteGraph,
    labels::AbstractVector,
    morphisms::AbstractVector,
    objects::AbstractVector) where {T₁, T₂, T₃}

    n₁ = nv₁(factorgraph)
    n₂ = nv₂(factorgraph)

    @assert n₁ == length(morphisms)
    @assert n₂ == length(labels) 
    @assert n₂ == length(objects)

    scopes = [Int[] for _ in 1:n₁]
    vvll = [Int[] for _ in 1:n₂]

    for i in edges(factorgraph)
        f = src(factorgraph, i)
        v = tgt(factorgraph, i)

        push!(scopes[f], v)
        insertsorted!(vvll[v], f)
    end

    factors = Vector{Factor{false, T₂, T₃}}(undef, n₁)
    graph = Graphs.Graph(n₂)

    for f in 1:n₁
        vars = scopes[f]
        n = length(vars)

        for i₁ in 2:n, i₂ in 1:i₁ - 1
            Graphs.add_edge!(graph, vars[i₁], vars[i₂])
        end

        hom = morphisms[f]
        obs = objects[vars]

        factors[f] = Factor{false}(hom, obs, vars)
    end

    labels = Labels{T₁}(labels)
    GraphicalModel(labels, factors, graph, vvll)
end


function GraphicalModel{T₁, T₂, T₃}(network::BayesNets.BayesNet) where {T₁, T₂, T₃}
    n = length(network)

    labels = Labels{T₁}(names(network))
    factors = Vector{Factor{false, T₂, T₃}}(undef, n)
    graph = Graphs.Graph{Int}(n)
    vvll = [[i] for i in 1:n]

    for i in 1:n
        cpd = network.cpds[i]
        parents =  map(l -> network.name_to_index[l], BayesNets.parents(cpd))
        m = length(parents)

        hom = GaussianSystem(cpd)
        obs = ones(Int, m + 1)
        vars = [parents; i]

        factors[i] = Factor{false}(hom, obs, vars)

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

            hom = fac.hom
            obs = fac.obs
            vars = replace(fac.vars, length(vvll) => v)

            factors[f] = Factor{false}(hom, obs, vars)
        end

        delete!(labels, l)
        Graphs.rem_vertex!(graph, v)
    
        vvll[v] = vvll[end]
        pop!(vvll)
    end

    GraphicalModel(labels, factors, graph, vvll)
end

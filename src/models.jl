# An undirected graphical model.
mutable struct GraphicalModel{T₁, T₂, T₃}
    labels::Labels{T₁}
    factors::Vector{Factor{T₂, T₃}}
    graph::Graphs.Graph{Int}
    vvll::Vector{Vector{Int}}
end


function GraphicalModel{T₁, T₂, T₃}(
    fg::AbstractUndirectedBipartiteGraph,
    homs::AbstractVector,
    obs::AbstractVector,
    labels::AbstractVector) where {T₁, T₂, T₃}

    @assert nv₁(fg) == length(homs)
    @assert nv₂(fg) == length(obs)

    scopes = [Int[] for _ in vertices₁(fg)]
    vvll = [Int[] for _ in vertices₂(fg)]

    for i in edges(fg)
        f = src(fg, i)
        v = tgt(fg, i)

        push!(scopes[f], v)
        push!(vvll[v], f)
    end

    labels = Labels{T₁}(labels)
    factors = Vector{Factor{T₂, T₃}}(undef, nv₁(fg))
    graph = Graphs.Graph(nv₂(fg))

    for (f, vs) in enumerate(scopes)
        factors[f] = Factor(homs[f], obs[vs], vs)
        n = length(vs)

        for i₁ in 2:n, i₂ in 1:i₁ - 1
            Graphs.add_edge!(graph, vs[i₁], vs[i₂])
        end
    end

    GraphicalModel(labels, factors, graph, vvll)
end


function GraphicalModel{T₁, T₂, T₃}(bn::BayesNets.BayesNet) where {T₁, T₂, T₃}
    n = length(bn)

    labels = Labels{T₁}(names(bn))
    factors = Vector{Factor{T₂, T₃}}(undef, n)
    graph = Graphs.Graph{Int}(n)
    vvll = [[i] for i in 1:n]

    for i in 1:n
        cpd = bn.cpds[i]
        pas = [bn.name_to_index[l] for l in BayesNets.parents(cpd)] 
        m = length(pas) 
  
        hom = GaussianSystem(cpd)
        obs = ones(Int, m + 1)
        vars = [pas; i]

        factors[i] = Factor(hom, obs, vars)

        for j₁ in 1:m
            Graphs.add_edge!(graph, pas[j₁], i)
            push!(vvll[pas[j₁]], i)

            for j₂ in 1:j₁ - 1
                Graphs.add_edge!(graph, pas[j₁], pas[j₂])
            end
        end
    end

    GraphicalModel(labels, factors, graph, vvll)
end


function Base.copy(model::GraphicalModel)
    labels = copy(model.labels)
    factors = copy(model.factors)
    graph = copy(model.graph)
    vvll = deepcopy(model.vvll)

    GraphicalModel(labels, factors, graph, vvll)
end


function observe!(model::GraphicalModel, context::Pair)
    l, hom = context

    v = model.labels.index[l]

    for f in model.vvll[v]
        model.factors[f] = observe(model.factors[f], v => hom)
    end

    u = length(model.labels)

    model.vvll[v] = model.vvll[u]

    for f in model.vvll[v]
        fac = model.factors[f]
        model.factors[f] = Factor(fac.hom, fac.obs, replace(fac.vars, u => v))
    end

    delete!(model.labels, l)
    Graphs.rem_vertex!(model.graph, v)
    pop!(model.vvll)
end


function observe!(model::GraphicalModel, context::AbstractDict)
    for (l, hom) in context
        observe!(model, l => hom)
    end
end

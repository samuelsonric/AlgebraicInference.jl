# An undirected graphical model.
mutable struct GraphicalModel{T₁, T₂, T₃}
    labels::Labels{T₁}
    factors::Vector{Factor{T₂, T₃}}
    graph::Graphs.Graph{Int}
    v_to_fs::Vector{Vector{Int}}
end


function GraphicalModel{T₁, T₂, T₃}(
    fg::AbstractUndirectedBipartiteGraph,
    homs::AbstractVector,
    obs::AbstractVector) where {T₁, T₂, T₃}

    @assert nv₁(fg) == length(homs)
    @assert nv₂(fg) == length(obs)

    f_to_vs = [Int[] for _ in vertices₁(fg)]
    v_to_fs = [Int[] for _ in vertices₂(fg)]

    for i in edges(fg)
        f = src(fg, i)
        v = tgt(fg, i)

        push!(f_to_vs[f], v)
        push!(v_to_fs[v], f)
    end

    labels = Labels{T₁}(vertices₂(fg))
    factors = Vector{Factor{T₂, T₃}}(undef, nv₁(fg))
    graph = Graphs.Graph(nv₂(fg))

    for (f, vs) in enumerate(f_to_vs)
        factors[f] = Factor(homs[f], obs[vs], vs)
        n = length(vs)

        for i₁ in 2:n, i₂ in 1:i₁ - 1
            Graphs.add_edge!(graph, vs[i₁], vs[i₂])
        end
    end

    GraphicalModel(labels, factors, graph, v_to_fs)
end


function GraphicalModel{T₁, T₂, T₃}(bn::BayesNet) where {T₁, T₂, T₃}
    n = length(bn)

    labels = Labels{T₁}(names(bn))
    factors = Vector{Factor{T₂, T₃}}(undef, n)
    graph = Graphs.Graph{Int}(n)
    v_to_fs = [[i] for i in 1:n]

    for i in 1:n
        cpd = bn.cpds[i]
        pas = [bn.name_to_index[l] for l in parents(cpd)] 
        m = length(pas) 
  
        obs = ones(Int, m + 1)
        vars = [pas; i]
        factors[i] = Factor(cpd, obs, vars)

        for j₁ in 1:m
            Graphs.add_edge!(graph, pas[j₁], i)
            push!(v_to_fs[pas[j₁]], i)

            for j₂ in 1:j₁ - 1
                Graphs.add_edge!(graph, pas[j₁], pas[j₂])
            end
        end
    end

    GraphicalModel(labels, factors, graph, v_to_fs)
end


function Base.copy(model::GraphicalModel)
    labels = copy(model.labels)
    factors = copy(model.factors)
    graph = copy(model.graph)
    v_to_fs = deepcopy(model.v_to_fs)

    GraphicalModel(labels, factors, graph, v_to_fs)
end


function observe!(model::GraphicalModel, evidence::Pair)
    l, hom = evidence

    v = model.labels.index[l]

    for f in model.v_to_fs[v]
        model.factors[f] = observe(model.factors[f], hom, v)
    end

    u = length(model.labels)

    model.v_to_fs[v] = model.v_to_fs[u]

    for f in model.v_to_fs[v]
        fac = model.factors[f]
        model.factors[f] = Factor(fac.hom, fac.obs, replace(fac.vars, u => v))
    end

    delete!(model.labels, l)
    Graphs.rem_vertex!(model.graph, v)
    pop!(model.v_to_fs)
end


function observe!(model::GraphicalModel, evidence::AbstractDict)
    for (l, hom) in evidence
        observe!(model, l => hom)
    end
end

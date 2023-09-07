"""
    InferenceProblem{T₁, T₂, T₃, T₄}

An inference problem computes the conditional distribution of ``X`` given ``Y = e``,
where ``X`` and ``Y`` are random variables whose joint probability distribution is
specified by a graphical model.
"""
mutable struct InferenceProblem{T₁, T₂, T₃, T₄}
    model::GraphicalModel{T₁, T₂, T₃}
    query::Vector{T₁}
    evidence::Dict{T₁, T₄}
end


function InferenceProblem{T₁, T₂, T₃, T₄}(
    uwd::AbstractUWD,
    hom_map::AbstractDict,
    ob_map::AbstractDict,
    evidence::AbstractDict=Dict();
    hom_attr::Symbol=:name,
    ob_attr::Symbol=:variable) where {T₁, T₂, T₃, T₄}

    homs = [hom_map[x] for x in subpart(uwd, hom_attr)]
    obs = [ob_map[x] for x in subpart(uwd, ob_attr)]

    InferenceProblem{T₁, T₂, T₃, T₄}(uwd, homs, obs, evidence)
end


function InferenceProblem{T₁, T₂, T₃, T₄}(
    uwd::AbstractUWD,
    homs::AbstractVector,
    obs::AbstractVector,
    evidence::AbstractDict=Dict()) where {T₁, T₂, T₃, T₄}

    e = evidence

    query = Vector{T₁}()
    evidence = Dict{T₁, T₄}()

    for i in ports(uwd; outer=true)
        v = junction(uwd, i; outer=true)

        if haskey(e, i)
            evidence[v] = e[i]
        else
            push!(query, v)
        end
    end

    fg = @migrate UndirectedBipartiteGraph uwd begin
        E  => Port
        V₁ => Box
        V₂ => Junction

        src => box
        tgt => junction
    end
    
    model = GraphicalModel{T₁, T₂, T₃}(fg, homs, obs)

    InferenceProblem{T₁, T₂, T₃, T₄}(model, query, evidence)
end


function InferenceProblem{T₁, T₂, T₃, T₄}(
    bn::BayesNet,
    query::AbstractVector,
    evidence::AbstractDict=Dict()) where {T₁, T₂, T₃, T₄}

    model = GraphicalModel{T₁, T₂, T₃}(bn)
    evidence = Dict(l => [v] for (l, v) in evidence)
   
    InferenceProblem{T₁, T₂, T₃, T₄}(model, query, evidence) 
end


"""
    InferenceProblem(
        uwd::AbstractUWD,
        hom_map::AbstractDict,
        ob_map::AbstractDict,
        evidence::AbstractDict=Dict();
        hom_attr::Symbol=:name,
        ob_attr::Symbol=:variable)

Construct an inference problem that performs undirected compositon.
"""
function InferenceProblem(
    uwd::AbstractUWD,
    hom_map::AbstractDict{<:Any, T₁},
    ob_map::AbstractDict{<:Any, T₂},
    evidence::AbstractDict{<:Any, T₃}=Dict();
    hom_attr::Symbol=:name,
    ob_attr::Symbol=:variable) where {T₁, T₂, T₃}

    InferenceProblem{Int, T₁, T₂, T₃}(uwd, hom_map, ob_map, evidence; hom_attr, ob_attr)
end


"""
    InferenceProblem(
        uwd::AbstractUWD,
        homs::AbstractVector,
        obs::AbstractVector,
        evidence::AbstractDict=Dict())

Construct an inference problem that performs undirected compositon.
"""
function InferenceProblem(
    uwd::AbstractUWD,
    homs::AbstractVector{T₁},
    obs::AbstractVector{T₂},
    evidence::AbstractDict{<:Any, T₃}=Dict()) where {T₁, T₂, T₃}

    InferenceProblem{Int, T₁, T₂, T₃}(uwd, homs, obs, evidence)
end


"""
    InferenceProblem(
        bn::BayesNet,
        query::AbstractVector,
        evidence::AbstractDict=Dict())

Construct an inference problem that queries a Bayesian network.
"""
function InferenceProblem(
    bn::BayesNet,
    query::AbstractVector,
    evidence::AbstractDict=Dict())

    InferenceProblem{Symbol, DenseCanonicalForm{Float64}, Int, Vector{Float64}}(bn, query, evidence)
end


"""
    solve(ip::InferenceProblem, alg::EliminationAlgorithm)

Solve an inference problem.
"""
CommonSolve.solve(ip::InferenceProblem, alg::EliminationAlgorithm)

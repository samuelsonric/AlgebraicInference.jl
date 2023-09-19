"""
    InferenceProblem{T₁, T₂, T₃, T₄}

An inference problem computes the conditional distribution of ``X`` given ``Y = e``,
where ``X`` and ``Y`` are random variables whose joint probability distribution is
specified by a graphical model.
"""
mutable struct InferenceProblem{T₁, T₂, T₃, T₄}
    model::GraphicalModel{T₁, T₂, T₃}
    query::Vector{T₁}
    context::Dict{T₁, T₄}
end


function InferenceProblem{T₁, T₂, T₃, T₄}(
    uwd::AbstractUWD,
    hom_map::AbstractDict,
    ob_map::AbstractDict,
    context::AbstractDict=Dict();
    hom_attr::Symbol=:name,
    ob_attr::Symbol=:junction_type,
    var_attr::Symbol=:variable) where {T₁, T₂, T₃, T₄}

    homs = [hom_map[x] for x in subpart(uwd, hom_attr)]
    obs = [ob_map[x] for x in subpart(uwd, ob_attr)]

    labels = subpart(uwd, var_attr)

    InferenceProblem{T₁, T₂, T₃, T₄}(uwd, homs, obs, context, labels)
end


function InferenceProblem{T₁, T₂, T₃, T₄}(
    uwd::AbstractUWD,
    homs::AbstractVector,
    obs::AbstractVector,
    context::AbstractDict=Dict(),
    labels::AbstractVector=junctions(uwd)) where {T₁, T₂, T₃, T₄}

    query = [
        labels[v] for v in subpart(uwd, :outer_junction)
        if !haskey(context, labels[v])]

    fg = @migrate UndirectedBipartiteGraph uwd begin
        E  => Port
        V₁ => Box
        V₂ => Junction

        src => box
        tgt => junction
    end
    
    model = GraphicalModel{T₁, T₂, T₃}(fg, homs, obs, labels)

    InferenceProblem{T₁, T₂, T₃, T₄}(model, query, context)
end


function InferenceProblem{T₁, T₂, T₃, T₄}(
    bn::BayesNets.BayesNet,
    query::AbstractVector,
    context::AbstractDict=Dict()) where {T₁, T₂, T₃, T₄}

    model = GraphicalModel{T₁, T₂, T₃}(bn)
    context = Dict(l => [v] for (l, v) in context)
   
    InferenceProblem{T₁, T₂, T₃, T₄}(model, query, context) 
end


"""
    InferenceProblem(
        uwd::RelationDiagram,
        hom_map::AbstractDict,
        ob_map::AbstractDict,
        evidence::AbstractDict=Dict();
        hom_attr::Symbol=:name,
        ob_attr::Symbol=:junction_type,
        var_attr::Symbol=:variable)

Construct an inference problem that performs undirected compositon.
"""
InferenceProblem(
    uwd::RelationDiagram,
    hom_map::AbstractDict,
    ob_map::AbstractDict,
    evidence::AbstractDict=Dict();
    hom_attr::Symbol=:name,
    ob_attr::Symbol=:junction_type,
    var_attr::Symbol=:variable)


function InferenceProblem(
    uwd::Union{TypedRelationDiagram{<:Any, <:Any, T₁}, UntypedRelationDiagram{<:Any, T₁}},
    hom_map::AbstractDict{<:Any, T₂},
    ob_map::AbstractDict{<:Any, T₃},
    context::AbstractDict{<:Any, T₄}=Dict();
    hom_attr::Symbol=:name,
    ob_attr::Symbol=:junction_type,
    var_attr::Symbol=:variable) where {T₁, T₂, T₃, T₄}    

    InferenceProblem{T₁, T₂, T₃, T₄}(uwd, hom_map, ob_map, context; hom_attr, ob_attr, var_attr)
end


"""
    InferenceProblem(
        bn::BayesNet,
        query::AbstractVector,
        evidence::AbstractDict=Dict())

Construct an inference problem that queries a Bayesian network.
"""
function InferenceProblem(
    bn::BayesNets.BayesNet,
    query::AbstractVector,
    context::AbstractDict=Dict())

    InferenceProblem{Symbol, DenseCanonicalForm{Float64}, Int, Vector{Float64}}(bn, query, context)
end


"""
    solve(
        problem::InferenceProblem,
        elalg::EliminationAlgorithm=MinFill()
        stype::SupernodeType=Node()
        atype::ArchitectureType=ShenoyShafer())

Solve an inference problem.
"""
CommonSolve.solve(
    problem::InferenceProblem,
    elalg::EliminationAlgorithm=MinFill(),
    stype::SupernodeType=Node(),
    atype::ArchitectureType=ShenoyShafer())

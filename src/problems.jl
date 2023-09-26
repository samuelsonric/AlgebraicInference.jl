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
    diagram::AbstractUWD,
    hom_map::AbstractDict,
    ob_map::AbstractDict;
    hom_attr::Symbol=:name,
    ob_attr::Symbol=:junction_type,
    var_attr::Symbol=:variable,
    check::Bool=true) where {T₁, T₂, T₃, T₄}

    labels = subpart(diagram, var_attr)
    morphisms = map(x -> hom_map[x], subpart(diagram, hom_attr))
    objects = map(x -> ob_map[x], subpart(diagram, ob_attr))

    InferenceProblem{T₁, T₂, T₃, T₄}(diagram, labels, morphisms, objects; check)
end


function InferenceProblem{T₁, T₂, T₃, T₄}(
    diagram::AbstractUWD,
    labels::AbstractVector,
    morphisms::AbstractVector,
    objects::AbstractVector;
    check::Bool=true) where {T₁, T₂, T₃, T₄}

    @assert !check || isvalid(diagram)

    factor_graph = @migrate UndirectedBipartiteGraph diagram begin
        E  => Port
        V₁ => Box
        V₂ => Junction

        src => box
        tgt => junction
    end
    
    model = GraphicalModel{T₁, T₂, T₃}(factor_graph, labels, morphisms, objects)
    query = labels[junction(diagram, :; outer=true)]
    context = Dict()

    InferenceProblem{T₁, T₂, T₃, T₄}(model, query, context)
end


function InferenceProblem{T₁, T₂, T₃, T₄}(
    network::BayesNets.BayesNet,
    query::AbstractVector,
    context::AbstractDict=Dict()) where {T₁, T₂, T₃, T₄}

    model = GraphicalModel{T₁, T₂, T₃}(network)
    context = Dict(l => [v] for (l, v) in context)
   
    InferenceProblem{T₁, T₂, T₃, T₄}(model, query, context) 
end


"""
    InferenceProblem(
        diagram::RelationDiagram,
        hom_map::AbstractDict,
        ob_map::AbstractDict;
        hom_attr::Symbol=:name,
        ob_attr::Symbol=:junction_type,
        var_attr::Symbol=:variable
        check::Bool=true)

Construct an inference problem that performs undirected compositon.
"""
InferenceProblem(
    diagram::RelationDiagram,
    hom_map::AbstractDict,
    ob_map::AbstractDict;
    hom_attr::Symbol=:name,
    ob_attr::Symbol=:junction_type,
    var_attr::Symbol=:variable,
    check::Bool=true)


function InferenceProblem(
    diagram::Union{TypedRelationDiagram{<:Any, <:Any, T₁}, UntypedRelationDiagram{<:Any, T₁}},
    hom_map::AbstractDict{<:Any, T₂},
    ob_map::AbstractDict{<:Any, T₃};
    hom_attr::Symbol=:name,
    ob_attr::Symbol=:junction_type,
    var_attr::Symbol=:variable,
    check::Bool=true) where {T₁, T₂, T₃} 

    InferenceProblem{T₁, T₂, T₃, ctxtype(T₂)}(
        diagram,
        hom_map,
        ob_map;
        hom_attr,
        ob_attr,
        var_attr,
        check)
end


"""
    InferenceProblem(
        network::BayesNet,
        query::AbstractVector,
        context::AbstractDict)

Construct an inference problem that queries a Bayesian network.
"""
function InferenceProblem(
    network::BayesNets.BayesNet,
    query::AbstractVector,
    context::AbstractDict)

    InferenceProblem{Symbol, DenseCanonicalForm{Float64}, Int, Vector{Float64}}(network, query, context)
end


"""
    solve(
        problem::InferenceProblem,
        elimination_algorithm::EliminationAlgorithm=MinFill()
        supernode_type::SupernodeType=Node()
        architecture_type::ArchitectureType=ShenoyShafer())

Solve an inference problem.
"""
CommonSolve.solve(
    problem::InferenceProblem,
    elimination_algorithm::EliminationAlgorithm=MinFill(),
    supernode_type::SupernodeType=Node(),
    architecture_type::ArchitectureType=ShenoyShafer())


# Add evidence to an inference problem.
function reduce_to_context(
    problem::InferenceProblem{T₁, T₂, T₃, T₄},
    context::AbstractDict) where {T₁, T₂, T₃, T₄}

    model = problem.model
    query = filter(l -> !haskey(context, l), problem.query)
    context = merge(problem.context, context)
    
    InferenceProblem{T₁, T₂, T₃, T₄}(model, query, context)
end

"""
    ArchitectureType

An algorithm that computes marginal distributions by passing messages over a join tree.
"""
abstract type ArchitectureType end


"""
    ShenoyShafer <: ArchitectureType

The Shenoy-Shafer architecture.
"""
struct ShenoyShafer <: ArchitectureType end


"""
    LauritzenSpiegelhalter <: ArchitectureType

The Lauritzen-Spiegelhalter architecture.
"""
struct LauritzenSpiegelhalter <: ArchitectureType end


# A mailbox in an architecture.
mutable struct Mailbox{T₁, T₂, T₃}
    factor::Union{Nothing, Factor{T₁, T₂}}
    cpd::Union{Nothing, CPD{T₃, T₂}}
    message_to_parent::Union{Nothing, Factor{T₁, T₂}}
    message_from_parent::Union{Nothing, Factor{T₁, T₂}}
end


# An architecture.
mutable struct Architecture{T₁, T₂, T₃, T₄}
    labels::Labels{T₁}
    factors::Vector{Factor{T₂, T₃}}
    tree::JoinTree
    assignments::Vector{Vector{Int}}
    mailboxes::Vector{Mailbox{T₂, T₃, T₄}}
    collect_phase_complete::Bool
end


# Construct an empty mailbox.
function Mailbox{T₁, T₂, T₃}() where {T₁, T₂, T₃}
    Mailbox{T₁, T₂, T₃}(nothing, nothing, nothing, nothing)
end


# Construct an architecture with empty mailboxes.
function Architecture(
    labels::Labels,
    factors::Vector{Factor{T₁, T₂}},
    tree::JoinTree,
    assignments::Vector{Vector{Int}}) where {T₁, T₂}

    T₃ = cpdtype(T₁)

    mailboxes = [Mailbox{T₁, T₂, T₃}() for _ in labels]
    collect_phase_complete = false

    Architecture(
        labels,
        factors,
        tree,
        assignments,
        mailboxes,
        collect_phase_complete)
end


# Construct an architecture.
function Architecture(
    model::GraphicalModel,
    elimination_algorithm::EliminationAlgorithm,
    supernode_type::SupernodeType)

    labels = model.labels
    factors = model.factors
    tree = JoinTree(model.graph, elimination_algorithm, supernode_type)

    vvll = deepcopy(model.vvll)
    assignments = Vector{Vector{Int}}(undef, length(tree))

    for n in tree.order
        node = IndexNode(tree, n)
        fs = Int[]

        for v in last(nodevalue(node)), f in vvll[v]
            push!(fs, f)
    
            for w in factors[f].vars
                if v != w
                    setdiff!(vvll[w], f)
                end
            end
        end

        assignments[node.index] = fs
    end

    Architecture(labels, factors, tree, assignments)
end


# Answer a query.
# Algorithm 4.2 in doi:10.1002/9781118010877.
function CommonSolve.solve!(
    architecture::Architecture,
    architecture_type::ShenoyShafer,
    query::AbstractVector)

    if !architecture.collect_phase_complete
        collect_phase!(architecture, architecture_type)
    end

    vars = [architecture.labels.index[l] for l in query]

    for n in architecture.tree.order
        node = IndexNode(architecture.tree, n)
        sep, res = nodevalue(node)

        if vars ⊆ [sep; res]
            distribute_phase!(architecture, architecture_type, node.index)

            mbx = mailbox(architecture, node.index)
            fac = combine(mbx.factor, mbx.message_from_parent)

            for child in children(node)
                mbx = mailbox(architecture, child.index)
                fac = combine(fac, mbx.message_to_parent)
            end 

            fac = project(fac, vars)
            hom = permute(fac, vars)
    
            return hom
        end 
    end

    error("Query not covered by join tree.")
end


# Answer a query.
# Algorithm 4.4 in doi:10.1002/9781118010877.
function CommonSolve.solve!(
    architecture::Architecture,
    architecture_type::LauritzenSpiegelhalter,
    query::AbstractVector)

    if !architecture.collect_phase_complete
        collect_phase!(architecture, architecture_type)
    end

    vars = [architecture.labels.index[l] for l in query]

    for n in architecture.tree.order
        node = IndexNode(architecture.tree, n)
        sep, res = nodevalue(node)

        if vars ⊆ [sep; res]
            distribute_phase!(architecture, architecture_type, node.index)

            mbx = mailbox(architecture, node.index)
            fac = combine(mbx.cpd, mbx.message_from_parent)

            fac = project(fac, vars)
            hom = permute(fac, vars)

            return hom
        end 
    end

    error("Query not covered by join tree.")
end


# Sample from an architecture.
function Base.rand(rng::AbstractRNG, architecture::Architecture)
    @assert architecture.collect_phase_complete

    m = length(architecture.labels)
    x = Vector{Vector{Float64}}(undef, m)

    for n in reverse(architecture.tree.order)
        node = IndexNode(architecture.tree, n)

        mbx = mailbox(architecture, node.index)
        rand!(rng, mbx.cpd, x)
    end

    Dict(zip(architecture.labels, x))
end


function Base.rand(arch::Architecture)
    rand(default_rng(), arch)
end


# Compute the mean of an architecture.
function Statistics.mean(architecture::Architecture)
    @assert architecture.collect_phase_complete

    m = length(architecture.labels)
    x = Vector{Vector{Float64}}(undef, m)

    for n in reverse(architecture.tree.order)
        node = IndexNode(architecture.tree, n)

        mbx = mailbox(architecture, node.index)
        mean!(mbx.cpd, x)
    end

    Dict(zip(architecture.labels, x))
end


# The collect phase of the Shenoy-Shafer architecture.
# Algorithm 4.1 in doi:10.1002/9781118010877.
function collect_phase!(
    architecture::Architecture{<:Any, T₁, T₂},
    architecture_type::ShenoyShafer) where {T₁, T₂}

    for n in architecture.tree.order
        node = IndexNode(architecture.tree, n)

        mbx = mailbox(architecture, node.index)
        mbx.factor = factor(architecture, node.index)
        msg = mbx.factor

        for child in children(node)
            mbx = mailbox(architecture, child.index)
            msg = combine(msg, mbx.message_to_parent)
        end

        mbx = mailbox(architecture, node.index)
        mbx.message_to_parent, mbx.cpd = disintegrate(msg, first(nodevalue(node)))
    end

    mbx = mailbox(architecture, rootindex(architecture.tree))
    mbx.message_from_parent = zero(Factor{T₁, T₂})
    architecture.collect_phase_complete = true
end


# The collect phase of the Lauritzen-Spiegelhalter architecture.
# Algorithm 4.3 in doi:10.1002/9781118010877.
function collect_phase!(
    architecture::Architecture{<:Any, T₁, T₂},
    architecture_type::LauritzenSpiegelhalter) where {T₁, T₂}

    for n in architecture.tree.order
        node = IndexNode(architecture.tree, n)

        msg = factor(architecture, node.index)

        for child in children(node)
            mbx = mailbox(architecture, child.index)
            msg = combine(msg, mbx.message_to_parent)
        end

        mbx = mailbox(architecture, node.index)
        mbx.message_to_parent, mbx.cpd = disintegrate(msg, first(nodevalue(node)))
    end

    mbx = mailbox(architecture, rootindex(architecture.tree))
    mbx.message_to_parent = nothing
    mbx.message_from_parent = zero(Factor{T₁, T₂})
    architecture.collect_phase_complete = true
end


# The distribute phase of the Shenoy-Shafer architecture. Only distributes from the root to
# node n.
# Algorithm 4.1 in doi:10.1002/9781118010877.
function distribute_phase!(
    architecture::Architecture,
    architecture_type::ShenoyShafer,
    n::Integer)

    node = IndexNode(architecture.tree, n)
    mbx = mailbox(architecture, node.index)

    ancestors = Int[]

    while !isroot(node) && isnothing(mbx.message_from_parent)
        push!(ancestors, node.index)
        node = parent(node)
        mbx = mailbox(architecture, node.index)
    end

    for n in ancestors[end:-1:1]
        node = IndexNode(architecture.tree, n)
        prnt = parent(node)

        mbx = mailbox(architecture, prnt.index)
        msg = combine(mbx.factor, mbx.message_from_parent)

        for sibling in children(prnt)
            if node != sibling
                mbx = mailbox(architecture, sibling.index)        
                msg = combine(msg, mbx.message_to_parent)
            end
        end

        mbx = mailbox(architecture, node.index)
        mbx.message_from_parent = project(msg, first(nodevalue(node)))
    end
end


# The distribute phase of the Lauritzen-Spiegelhalter architecture. Only distributes from
# the root to node n.
# Algorithm 4.3 in doi:10.1002/9781118010877.
function distribute_phase!(
    architecture::Architecture,
    architecture_type::LauritzenSpiegelhalter,
    n::Integer)

    node = IndexNode(architecture.tree, n)
    mbx = mailbox(architecture, node.index)

    ancestors = Int[]

    while !isroot(node) && isnothing(mbx.message_from_parent)
        push!(ancestors, node.index)
        node = parent(node)
        mbx = mailbox(architecture, node.index)
    end

    for n in ancestors[end:-1:1]
        node = IndexNode(architecture.tree, n)
        prnt = parent(node)

        mbx = mailbox(architecture, prnt.index)
        msg = combine(mbx.cpd, mbx.message_from_parent)

        mbx = mailbox(architecture, node.index)
        mbx.message_to_parent = nothing
        mbx.message_from_parent = project(msg, first(nodevalue(node)))
    end
end


# Get the mailbox containing
# μ: n → pa(n)
# and
# μ: pa(n) → n
function mailbox(arch::Architecture, n::Integer)
    arch.mailboxes[n]
end


# Compute the join tree factor
# ψₙ
function factor(architecture::Architecture{<:Any, T₁, T₂}, n::Integer) where {T₁, T₂}
    fac = zero(Factor{T₁, T₂})

    for f in architecture.assignments[n]
        fac = combine(fac, architecture.factors[f])
    end

    fac
end

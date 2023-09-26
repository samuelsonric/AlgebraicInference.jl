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


"""
    HUGIN <: ArchitectureType

The HUGIN architecture.
"""
struct HUGIN <: ArchitectureType end


"""
    Idempotent <: ArchitectureType

The idempotent architecture.
"""
struct Idempotent <: ArchitectureType end


"""
    AncestralSampler <: ArchitectureType

A type for sampling graphical models.
"""
struct AncestralSampler <: ArchitectureType end


# A mailbox in an architecture.
mutable struct Mailbox{T₁, T₂, T₃}
    factor::Union{Nothing, Factor{true, T₁, T₂}}
    cpd::Union{Nothing, CPD{T₃, T₂}}
    message_to_parent::Union{Nothing, Factor{true, T₁, T₂}}
    message_from_parent::Union{Nothing, Factor{true, T₁, T₂}}
    distribute_phase_complete::Bool
end


# An architecture.
mutable struct Architecture{T₁, T₂, T₃, T₄, T₅}
    labels::Labels{T₂}
    factors::Vector{Factor{true, T₃, T₄}}
    tree::JoinTree
    assignments::Vector{Vector{Int}}
    mailboxes::Vector{Mailbox{T₃, T₄, T₅}}
    collect_phase_complete::Bool
end


# Construct an empty mailbox.
function Mailbox{T₁, T₂, T₃}() where {T₁, T₂, T₃}
    Mailbox{T₁, T₂, T₃}(nothing, nothing, nothing, nothing, false)
end


# Construct an architecture with empty mailboxes.
function Architecture(
    labels::Labels{T₁},
    factors::Vector{Factor{true, T₂, T₃}},
    tree::JoinTree,
    assignments::Vector{Vector{Int}},
    architecture_type::ArchitectureType) where {T₁, T₂, T₃}

    T₄ = cpdtype(T₂)

    mailboxes = [Mailbox{T₂, T₃, T₄}() for _ in labels]
    collect_phase_complete = false

    Architecture{architecture_type, T₁, T₂, T₃, T₄}(
        labels,
        factors,
        tree,
        assignments,
        mailboxes,
        collect_phase_complete)
end


# Construct an architecture.
function Architecture(
    model::GraphicalModel{<:Any, T₁, T₂},
    elimination_algorithm::EliminationAlgorithm,
    supernode_type::SupernodeType,
    architecture_type::ArchitectureType) where {T₁, T₂}

    labels = model.labels
    tree = JoinTree(model.graph, elimination_algorithm, supernode_type)
    factors = Vector{Factor{true, T₁, T₂}}(undef, length(model.factors))

    for (f, fac) in enumerate(model.factors)
        factors[f] = sort(fac)
    end

    vvll = deepcopy(model.vvll)
    assignments = Vector{Vector{Int}}(undef, length(tree))

    for n in tree.order
        node = IndexNode(tree, n)
        fs = Int[]

        for v in last(nodevalue(node)), f in vvll[v]
            insertsorted!(fs, f)
    
            for w in factors[f].vars
                if v != w
                    deletesorted!(vvll[w], f)
                end
            end
        end

        assignments[node.index] = fs
    end

    Architecture(labels, factors, tree, assignments, architecture_type)
end


# Answer a query.
# Algorithm 4.2 in doi:10.1002/9781118010877.
function CommonSolve.solve!(architecture::Architecture{ShenoyShafer()}, query::AbstractVector)
    if !architecture.collect_phase_complete
        collect_phase!(architecture)
    end

    vars = query_variables(architecture, query)
    node = covering_node(architecture, vars)
    distribute_phase!(architecture, node.index)

    mbx = mailbox(architecture, node.index)
    fac = combine(mbx.factor, mbx.message_from_parent)

    for child in children(node)
        mbx = mailbox(architecture, child.index)
        fac = combine(fac, mbx.message_to_parent)
    end 

    fac = project(fac, vars)
    hom = permute(fac, vars)

    hom
end


# Answer a query.
# Algorithm 4.4 in doi:10.1002/9781118010877.
function CommonSolve.solve!(
    architecture::Union{
        Architecture{LauritzenSpiegelhalter()},
        Architecture{HUGIN()},
        Architecture{Idempotent()}},
    query::AbstractVector)

    if !architecture.collect_phase_complete
        collect_phase!(architecture)
    end

    vars = query_variables(architecture, query)
    node = covering_node(architecture, vars)
    distribute_phase!(architecture, node.index)

    mbx = mailbox(architecture, node.index)
    fac = mbx.factor

    fac = project(fac, vars)
    hom = permute(fac, vars)

    hom
end


function CommonSolve.solve!(architecture::Architecture{AncestralSampler()})
    if !architecture.collect_phase_complete
        collect_phase!(architecture)
    end
end


# Sample from an architecture.
function Base.rand(
    rng::AbstractRNG,
    architecture::Architecture{AncestralSampler(), <:Any, T},
    query::AbstractVector) where T

    @assert architecture.collect_phase_complete

    m = length(architecture.labels)
    x = Vector{ctxtype(T)}(undef, m)

    for n in reverse(architecture.tree.order)
        node = IndexNode(architecture.tree, n)
        mbx = mailbox(architecture, node.index)
        cpdrand!(rng, mbx.cpd, x)
    end
    
    vars = query_variables(architecture, query)

    ctxcat(T, x[vars])
end


# Compute the mean of an architecture.
function Statistics.mean(
    architecture::Architecture{AncestralSampler(), <:Any, T},
    query::AbstractVector) where T

    @assert architecture.collect_phase_complete

    m = length(architecture.labels)
    x = Vector{ctxtype(T)}(undef, m)

    for n in reverse(architecture.tree.order)
        node = IndexNode(architecture.tree, n)
        mbx = mailbox(architecture, node.index)
        cpdmean!(mbx.cpd, x)
    end

    vars = query_variables(architecture, query)

    ctxcat(T, x[vars])
end


# The collect phase of the Shenoy-Shafer architecture.
# Algorithm 4.1 in doi:10.1002/9781118010877.
function collect_phase!(architecture::Architecture{ShenoyShafer(), <:Any, T₁, T₂}) where {
    T₁, T₂}

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
        mbx.message_to_parent = project(msg, first(nodevalue(node)))
    end

    mbx = mailbox(architecture, rootindex(architecture.tree))
    mbx.message_from_parent = unit(Factor{true, T₁, T₂})
    architecture.collect_phase_complete = true
end


# The collect phase of the Lauritzen-Spiegelhalter architecture.
# Algorithm 4.3 in doi:10.1002/9781118010877.
function collect_phase!(architecture::Architecture{LauritzenSpiegelhalter()})
    for n in architecture.tree.order
        node = IndexNode(architecture.tree, n)
        mbx = mailbox(architecture, node.index)
        mbx.factor = factor(architecture, node.index)
    end

    for n in architecture.tree.order[1:end - 1]
        node = IndexNode(architecture.tree, n)
        mbx = mailbox(architecture, node.index)
        msg = project(mbx.factor, first(nodevalue(node)))
        mbx.factor = combine(mbx.factor, invert(msg))

        node = parent(node)
        mbx = mailbox(architecture, node.index)
        mbx.factor = combine(mbx.factor, msg)
    end

    architecture.collect_phase_complete = true
end


# The collect phase of the HUGIN architecture.
# Algorithm 4.5 in doi:10.1002/9781118010877.
function collect_phase!(architecture::Architecture{HUGIN()})
    for n in architecture.tree.order
        node = IndexNode(architecture.tree, n)
        mbx = mailbox(architecture, node.index)
        mbx.factor = factor(architecture, node.index)
    end

    for n in architecture.tree.order[1:end - 1]
        node = IndexNode(architecture.tree, n)
        mbx = mailbox(architecture, node.index)
        msg = project(mbx.factor, first(nodevalue(node)))
        mbx.message_to_parent = msg

        node = parent(node)
        mbx = mailbox(architecture, node.index)
        mbx.factor = combine(mbx.factor, msg)
    end

    architecture.collect_phase_complete = true
end


# The collect phase of the idempotent architecture.
# Algorithm 4.6 in doi:10.1002/9781118010877.
function collect_phase!(architecture::Architecture{Idempotent()})
    for n in architecture.tree.order
        node = IndexNode(architecture.tree, n)
        mbx = mailbox(architecture, node.index)
        mbx.factor = factor(architecture, node.index)
    end

    for n in architecture.tree.order[1:end - 1]
        node = IndexNode(architecture.tree, n)
        mbx = mailbox(architecture, node.index)
        msg = project(mbx.factor, first(nodevalue(node)))

        node = parent(node)
        mbx = mailbox(architecture, node.index)
        mbx.factor = combine(mbx.factor, msg)
    end

    architecture.collect_phase_complete = true
end


function collect_phase!(architecture::Architecture{AncestralSampler()})
    for n in architecture.tree.order
        node = IndexNode(architecture.tree, n)
        mbx = mailbox(architecture, node.index)
        mbx.factor = factor(architecture, node.index)
    end

    for n in architecture.tree.order[1:end - 1]
        node = IndexNode(architecture.tree, n)
        mbx = mailbox(architecture, node.index)
        msg, mbx.cpd = disintegrate(mbx.factor, first(nodevalue(node)))
        mbx.factor = nothing

        node = parent(node)
        mbx = mailbox(architecture, node.index)
        mbx.factor = combine(mbx.factor, msg)
    end

    node = IndexNode(architecture.tree)
    mbx = mailbox(architecture, node.index)
    mbx.cpd = last(disintegrate(mbx.factor, first(nodevalue(node))))
    mbx.factor = nothing
    architecture.collect_phase_complete = true
end


# The distribute phase of the Shenoy-Shafer architecture. Only distributes from the root to
# node n.
# Algorithm 4.1 in doi:10.1002/9781118010877.
function distribute_phase!(architecture::Architecture{ShenoyShafer()}, n::Integer)
    for n in reverse(ancestors(architecture, n))
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
        mbx.distribute_phase_complete = true
    end
end


# The distribute phase of the Lauritzen-Spiegelhalter architecture. Only distributes from
# the root to node n.
# Algorithm 4.3 in doi:10.1002/9781118010877.
function distribute_phase!(
    architecture::Union{
        Architecture{LauritzenSpiegelhalter()},
        Architecture{Idempotent()}},
    n::Integer)

    for n in reverse(ancestors(architecture, n))
        node = IndexNode(architecture.tree, n)
        prnt = parent(node)

        mbx = mailbox(architecture, prnt.index)
        msg = mbx.factor

        mbx = mailbox(architecture, node.index)
        msg = project(msg, first(nodevalue(node)))
        mbx.factor = combine(mbx.factor, msg)
        mbx.distribute_phase_complete = true
    end
end


# The distribute phase of the HUGIN architecture. Only distributes from
# the root to node n.
# Algorithm 4.5 in doi:10.1002/9781118010877.
function distribute_phase!(architecture::Architecture{HUGIN()}, n::Integer)
    for n in reverse(ancestors(architecture, n))
        node = IndexNode(architecture.tree, n)
        prnt = parent(node)

        mbx = mailbox(architecture, prnt.index)
        msg = mbx.factor

        mbx = mailbox(architecture, node.index)
        msg = project(msg, first(nodevalue(node)))
        msg = combine(msg, invert(mbx.message_to_parent))
        mbx.factor = combine(mbx.factor, msg)
        mbx.message_to_parent = nothing
        mbx.distribute_phase_complete = true
    end
end


# Get the mailbox containing
# μ: n → pa(n)
# and
# μ: pa(n) → n
function mailbox(architecture::Architecture, n::Integer)
    architecture.mailboxes[n]
end


# Compute the join tree factor
# ψₙ
function factor(architecture::Architecture{<:Any, <:Any, T₁, T₂}, n::Integer) where {T₁, T₂}
    fac = unit(Factor{true, T₁, T₂})

    for f in architecture.assignments[n]
        fac = combine(fac, architecture.factors[f])
    end

    fac
end


# Get the variables corresponding to a query.
function query_variables(architecture::Architecture, query::AbstractVector)
    [architecture.labels.index[l] for l in query]
end


# Find a node that covers a collection of variables.
function covering_node(architecture::Architecture, vars::AbstractVector)
    for n in architecture.tree.order
        node = IndexNode(architecture.tree, n)
        sep, res = nodevalue(node)

        if vars ⊆ [sep; res]
            return node
        end 
    end

    error("Query not covered by join tree.")
end


# Compute the ancestors of node n.
function ancestors(architecture::Architecture, n::Integer)
    node = IndexNode(architecture.tree, n)
    mbx = mailbox(architecture, node.index)

    ancestors = Int[]

    while !isroot(node) && !mbx.distribute_phase_complete
        push!(ancestors, node.index)
        node = parent(node)
        mbx = mailbox(architecture, node.index)
    end

    ancestors
end

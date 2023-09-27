"""
    ArchitectureType

A message-passing algorithm. The options are
- [`ShenoyShafer`](@ref)
- [`LauritzenSpiegelhalter`](@ref)
- [`HUGIN`](@ref)
- [`Idempotent`](@ref)

There is one additional type, which you can use for sampling from a graphical model:
- [`AncestralSamper`](@ref)
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

    vvll = deepcopy(model.vvll)
 
    labels = model.labels
    tree = JoinTree(model.graph, elimination_algorithm, supernode_type)
    factors = Vector{Factor{true, T₁, T₂}}(undef, length(model.factors))
    assignments = Vector{Vector{Int}}(undef, length(tree))
 
    for f in eachindex(model.factors)
        factors[f] = sort(model.factors[f])
    end

    for n in tree.order
        residual = last(tree[n])
        assignments[n] = Int[]

        for v in residual, f in vvll[v]
            insertsorted!(assignments[n], f)
        end

        for f in assignments[n], v in factors[f].vars
            deletesorted!(vvll[v], f)
        end
    end

    Architecture(labels, factors, tree, assignments, architecture_type)
end


# Answer a query.
# Algorithm 4.2 in doi:10.1002/9781118010877.
function CommonSolve.solve!(architecture::Architecture{ShenoyShafer()}, query::AbstractVector)
    if !architecture.collect_phase_complete
        collect_phase!(architecture)
    end

    variables = getvariables(architecture, query)
    n = findnode(architecture, variables)

    distribute_phase!(architecture, n)
    mailbox = getmailbox(architecture, n)
    factor = combine(mailbox.factor, mailbox.message_from_parent)

    for m in childindices(architecture, n)
        mailbox = getmailbox(architecture, m)
        factor = combine(factor, mailbox.message_to_parent)
    end 

    factor = project(factor, variables)
    permute(factor, variables)
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

    variables = getvariables(architecture, query)
    n = findnode(architecture, variables)

    distribute_phase!(architecture, n)
    mailbox = getmailbox(architecture, n)
    factor = project(mailbox.factor, variables)
    permute(factor, variables)
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

    order = getorder(architecture)
    samples = Vector{ctxtype(T)}(undef, nvariables(architecture))

    for n in reverse(order)
        mailbox = getmailbox(architecture, n)
        cpdrand!(rng, mailbox.cpd, samples)
    end
    
    variables = getvariables(architecture, query)
    ctxcat(T, samples[variables])
end


# Compute the mean of an architecture.
function Statistics.mean(
    architecture::Architecture{AncestralSampler(), <:Any, T},
    query::AbstractVector) where T

    @assert architecture.collect_phase_complete

    order = getorder(architecture)
    means = Vector{ctxtype(T)}(undef, nvariables(architecture))

    for n in reverse(order)
        mailbox = getmailbox(architecture, n)
        cpdmean!(mailbox.cpd, means)
    end

    variables = getvariables(architecture, query)
    ctxcat(T, means[variables])
end


# The collect phase of the Shenoy-Shafer architecture.
# Algorithm 4.1 in doi:10.1002/9781118010877.
function collect_phase!(architecture::Architecture{ShenoyShafer(), <:Any, T₁, T₂}) where {T₁, T₂}
    order = getorder(architecture)

    for n in order
        mailbox = getmailbox(architecture, n)
        mailbox.factor = getfactor(architecture, n)
        message = mailbox.factor

        for m in childindices(architecture, n)
            mailbox = getmailbox(architecture, m)
            message = combine(message, mailbox.message_to_parent)
        end

        seperator = getseperator(architecture, n)
        mailbox = getmailbox(architecture, n)
        mailbox.message_to_parent = project(message, seperator)
    end

    n = rootindex(architecture)
    mailbox = getmailbox(architecture, n)
    mailbox.message_from_parent = unit(Factor{true, T₁, T₂})

    architecture.collect_phase_complete = true
end


# The collect phase of the Lauritzen-Spiegelhalter architecture.
# Algorithm 4.3 in doi:10.1002/9781118010877.
function collect_phase!(architecture::Architecture{LauritzenSpiegelhalter()})
    order = getorder(architecture)

    for n in order
        mailbox = getmailbox(architecture, n)
        mailbox.factor = getfactor(architecture, n)
    end

    for n in order[1:end - 1]
        seperator = getseperator(architecture, n)
        mailbox = getmailbox(architecture, n)
        message = project(mailbox.factor, seperator)
        mailbox.factor = combine(mailbox.factor, invert(message))

        n = parentindex(architecture, n)
        mailbox = getmailbox(architecture, n)
        mailbox.factor = combine(mailbox.factor, message)
    end

    architecture.collect_phase_complete = true
end


# The collect phase of the HUGIN architecture.
# Algorithm 4.5 in doi:10.1002/9781118010877.
function collect_phase!(architecture::Architecture{HUGIN()})
    order = getorder(architecture)

    for n in order
        mailbox = getmailbox(architecture, n)
        mailbox.factor = getfactor(architecture, n)
    end

    for n in order[1:end - 1]
        seperator = getseperator(architecture, n)
        mailbox = getmailbox(architecture, n)
        message = project(mailbox.factor, seperator)
        mailbox.message_to_parent = message

        n = parentindex(architecture, n)
        mailbox = getmailbox(architecture, n)
        mailbox.factor = combine(mailbox.factor, message)
    end

    architecture.collect_phase_complete = true
end


# The collect phase of the idempotent architecture.
# Algorithm 4.6 in doi:10.1002/9781118010877.
function collect_phase!(architecture::Architecture{Idempotent()})
    order = getorder(architecture)

    for n in order
        mailbox = getmailbox(architecture, n)
        mailbox.factor = getfactor(architecture, n)
    end

    for n in order[1:end - 1]
        seperator = getseperator(architecture, n)
        mailbox = getmailbox(architecture, n)
        message = project(mailbox.factor, seperator)

        n = parentindex(architecture, n)
        mailbox = getmailbox(architecture, n)
        mailbox.factor = combine(mailbox.factor, message)
    end

    architecture.collect_phase_complete = true
end


function collect_phase!(architecture::Architecture{AncestralSampler()})
    order = getorder(architecture)

    for n in order
        mailbox = getmailbox(architecture, n)
        mailbox.factor = getfactor(architecture, n)
    end

    for n in order[1:end - 1]
        seperator = getseperator(architecture, n)
        mailbox = getmailbox(architecture, n)
        message, mailbox.cpd = disintegrate(mailbox.factor, seperator)
        mailbox.factor = nothing

        n = parentindex(architecture, n)
        mailbox = getmailbox(architecture, n)
        mailbox.factor = combine(mailbox.factor, message)
    end

    n = rootindex(architecture)
    seperator = getseperator(architecture, n)
    mailbox = getmailbox(architecture, n)
    _, mailbox.cpd = disintegrate(mailbox.factor, seperator)
    mailbox.factor = nothing

    architecture.collect_phase_complete = true
end


# The distribute phase of the Shenoy-Shafer architecture. Only distributes from the root to
# node n.
# Algorithm 4.1 in doi:10.1002/9781118010877.
function distribute_phase!(architecture::Architecture{ShenoyShafer()}, n::Integer)
    for n in reverse(getancestors(architecture, n))
        m = parentindex(architecture, n)
        mailbox = getmailbox(architecture, m)
        message = combine(mailbox.factor, mailbox.message_from_parent)

        for l in childindices(architecture, m)
            if l != n
                mailbox = getmailbox(architecture, l)
                message = combine(message, mailbox.message_to_parent)
            end
        end

        seperator = getseperator(architecture, n)
        mailbox = getmailbox(architecture, n)
        mailbox.message_from_parent = project(message, seperator)
        mailbox.distribute_phase_complete = true
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

    for n in reverse(getancestors(architecture, n))
        m = parentindex(architecture, n)
        mailbox = getmailbox(architecture, m)
        message = mailbox.factor

        seperator = getseperator(architecture, n)
        mailbox = getmailbox(architecture, n)
        message = project(message, seperator)
        mailbox.factor = combine(mailbox.factor, message)
        mailbox.distribute_phase_complete = true
    end
end


# The distribute phase of the HUGIN architecture. Only distributes from
# the root to node n.
# Algorithm 4.5 in doi:10.1002/9781118010877.
function distribute_phase!(architecture::Architecture{HUGIN()}, n::Integer)
    for n in reverse(getancestors(architecture, n))
        m = parentindex(architecture, n)
        mailbox = getmailbox(architecture, m)
        message = mailbox.factor

        seperator = getseperator(architecture, n)
        mailbox = getmailbox(architecture, n)
        message = project(message, seperator)
        message = combine(message, invert(mailbox.message_to_parent))
        mailbox.factor = combine(mailbox.factor, message)
        mailbox.message_to_parent = nothing
        mailbox.distribute_phase_complete = true
    end
end


# Get the mailbox containing
# μ: n → pa(n)
# and
# μ: pa(n) → n
function getmailbox(architecture::Architecture, n::Integer)
    architecture.mailboxes[n]
end


function getseperator(architecture::Architecture, n::Integer)
    seperator, _ = nodevalue(architecture.tree, n)
    seperator
end


function getresidual(architecture::Architecture, n::Integer)
    _, residual = nodevalue(architecture.tree, n)
    residual
end


function getorder(architecture::Architecture)
    architecture.tree.order
end


# Compute the join tree factor
# ψₙ
function getfactor(architecture::Architecture{<:Any, <:Any, T₁, T₂}, n::Integer) where {T₁, T₂}
    factor = unit(Factor{true, T₁, T₂})

    for f in architecture.assignments[n]
        factor = combine(factor, architecture.factors[f])
    end

    factor
end


# Get the variables corresponding to a query.
function getvariables(architecture::Architecture, query::AbstractVector)
    [architecture.labels.index[l] for l in query]
end


function nvariables(architecture::Architecture)
    length(architecture.labels)
end


# Find a node that covers a collection of variables.
function findnode(architecture::Architecture, variables::AbstractVector)
    order = getorder(architecture)

    i = findfirst(order) do n
        seperator = getseperator(architecture, n)
        residual = getresidual(architecture, n)
        variables ⊆ [seperator; residual]
    end

    if isnothing(i)
        error("Query not covered by join tree.")
    else
        order[i]
    end
end


# Compute the ancestors of node n.
function getancestors(architecture::Architecture, n::Integer)
    mailbox = getmailbox(architecture, n)
    ancestors = Int[]

    while n != rootindex(architecture) && !mailbox.distribute_phase_complete
        push!(ancestors, n)
        n = parentindex(architecture, n)
        mailbox = getmailbox(architecture, n)
    end

    ancestors
end


##########################
# Indexed Tree interface #
##########################


function AbstractTrees.rootindex(architecture::Architecture)
    rootindex(architecture.tree)
end


function AbstractTrees.parentindex(architecture::Architecture, n::Integer)
    parentindex(architecture.tree, n)
end


function AbstractTrees.childindices(architecture::Architecture, n::Integer)
    childindices(architecture.tree, n)
end

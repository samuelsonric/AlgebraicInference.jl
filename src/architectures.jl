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
    elalg::EliminationAlgorithm,
    stype::SupernodeType)

    labels = model.labels
    factors = model.factors
    tree = JoinTree(model.graph, elalg, stype)

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
function CommonSolve.solve!(arch::Architecture, atype::ShenoyShafer, query)
    arch.collect_phase_complete || collect_phase!(arch, atype)

    vars = [arch.labels.index[l] for l in query]

    for n in arch.tree.order
        node = IndexNode(arch.tree, n)
        sep, res = nodevalue(node)

        if vars ⊆ [sep; res]
            distribute_phase!(arch, atype, node.index)

            mbx = mailbox(arch, node.index)
            fac = combine(mbx.factor, mbx.message_from_parent)

            for child in children(node)
                mbx = mailbox(arch, child.index)
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
function CommonSolve.solve!(arch::Architecture, atype::LauritzenSpiegelhalter, query)
    arch.collect_phase_complete || collect_phase!(arch, atype)

    vars = [arch.labels.index[l] for l in query]

    for n in arch.tree.order
        node = IndexNode(arch.tree, n)
        sep, res = nodevalue(node)

        if vars ⊆ [sep; res]
            distribute_phase!(arch, atype, node.index)

            mbx = mailbox(arch, node.index)
            fac = combine(mbx.cpd, mbx.message_from_parent)

            fac = project(fac, vars)
            hom = permute(fac, vars)

            return hom
        end 
    end

    error("Query not covered by join tree.")
end


# Sample from an architecture.
function Base.rand(rng::AbstractRNG, arch::Architecture)
    @assert arch.collect_phase_complete

    x = Vector{Vector{Float64}}(undef, length(arch.labels))

    for n in reverse(arch.tree.order)
        node = IndexNode(arch.tree, n)

        mbx = mailbox(arch, node.index)
        rand!(rng, mbx.cpd, x)
    end

    Dict(zip(arch.labels, x))
end


function Base.rand(arch::Architecture)
    rand(default_rng(), arch)
end


# Compute the mean of an architecture.
function Statistics.mean(arch::Architecture)
    @assert arch.collect_phase_complete

    x = Vector{Vector{Float64}}(undef, length(arch.labels))

    for n in reverse(arch.tree.order)
        node = IndexNode(arch.tree, n)

        mbx = mailbox(arch, node.index)
        mean!(mbx.cpd, x)
    end

    Dict(zip(arch.labels, x))
end


# The collect phase of the Shenoy-Shafer architecture.
# Algorithm 4.1 in doi:10.1002/9781118010877.
function collect_phase!(arch::Architecture{<:Any, T₁, T₂}, atype::ShenoyShafer) where {T₁, T₂}
    for n in arch.tree.order
        node = IndexNode(arch.tree, n)

        mbx = mailbox(arch, node.index)
        mbx.factor = factor(arch, node.index)
        msg = mbx.factor

        for child in children(node)
            mbx = mailbox(arch, child.index)
            msg = combine(msg, mbx.message_to_parent)
        end

        mbx = mailbox(arch, node.index)
        mbx.message_to_parent, mbx.cpd = disintegrate(msg, first(nodevalue(node)))
    end

    mbx = mailbox(arch, rootindex(arch.tree))
    mbx.message_from_parent = zero(Factor{T₁, T₂})
    arch.collect_phase_complete = true
end


# The collect phase of the Lauritzen-Spiegelhalter architecture.
# Algorithm 4.3 in doi:10.1002/9781118010877.
function collect_phase!(arch::Architecture{<:Any, T₁, T₂}, atype::LauritzenSpiegelhalter) where {T₁, T₂}
    for n in arch.tree.order
        node = IndexNode(arch.tree, n)

        msg = factor(arch, node.index)

        for child in children(node)
            mbx = mailbox(arch, child.index)
            msg = combine(msg, mbx.message_to_parent)
        end

        mbx = mailbox(arch, node.index)
        mbx.message_to_parent, mbx.cpd = disintegrate(msg, first(nodevalue(node)))
    end

    mbx = mailbox(arch, rootindex(arch.tree))
    mbx.message_to_parent = nothing
    mbx.message_from_parent = zero(Factor{T₁, T₂})
    arch.collect_phase_complete = true
end


# The distribute phase of the Shenoy-Shafer architecture.
# Algorithm 4.1 in doi:10.1002/9781118010877.
function distribute_phase!(arch::Architecture, atype::ShenoyShafer, n::Integer)
    node = IndexNode(arch.tree, n)
    mbx = mailbox(arch, node.index)

    ancestors = Int[]

    while !isroot(node) && isnothing(mbx.message_from_parent)
        push!(ancestors, node.index)
        node = parent(node)
        mbx = mailbox(arch, node.index)
    end

    for n in ancestors[end:-1:1]
        node = IndexNode(arch.tree, n)
        prnt = parent(node)

        mbx = mailbox(arch, prnt.index)
        msg = combine(mbx.factor, mbx.message_from_parent)

        for sibling in children(prnt)
            if node != sibling
                mbx = mailbox(arch, sibling.index)        
                msg = combine(msg, mbx.message_to_parent)
            end
        end

        mbx = mailbox(arch, node.index)
        mbx.message_from_parent = project(msg, first(nodevalue(node)))
    end
end


# The distribute phase of the Lauritzen-Spiegelhalter architecture.
# Algorithm 4.3 in doi:10.1002/9781118010877.
function distribute_phase!(arch::Architecture, atype::LauritzenSpiegelhalter, n::Integer)
    node = IndexNode(arch.tree, n)
    mbx = mailbox(arch, node.index)

    ancestors = Int[]

    while !isroot(node) && isnothing(mbx.message_from_parent)
        push!(ancestors, node.index)
        node = parent(node)
        mbx = mailbox(arch, node.index)
    end

    for n in ancestors[end:-1:1]
        node = IndexNode(arch.tree, n)
        prnt = parent(node)

        mbx = mailbox(arch, prnt.index)
        msg = combine(mbx.cpd, mbx.message_from_parent)

        mbx = mailbox(arch, node.index)
        mbx.message_to_parent = nothing
        mbx.message_from_parent = project(msg, first(nodevalue(node)))
    end
end


# Get the mailbox containing
# μ: n → pa(n)
# and
# μ: pa(n) → n
function mailbox(arch::Architecture, n::Int)
    arch.mailboxes[n]
end


# Compute the join tree factor
# ψₙ
function factor(arch::Architecture{<:Any, T₁, T₂}, n::Int) where {T₁, T₂}
    fac = zero(Factor{T₁, T₂})

    for f in arch.assignments[n]
        fac = combine(fac, arch.factors[f])
    end

    fac
end

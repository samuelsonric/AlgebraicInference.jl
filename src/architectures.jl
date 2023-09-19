"""
    ArchitectureType

An algorithm that computes marginals by passing messages over a join tree.
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


# A mailbox in the Shenoy-Shafer architecture.
mutable struct SSMailbox{T₁, T₂, T₃}
    factor::Union{Nothing, Factor{T₁, T₂}}
    message_to_parent::Union{Nothing, Factor{T₁, T₂}}
    message_from_parent::Union{Nothing, Factor{T₁, T₂}}
    cpd::Union{Nothing, CPD{T₃, T₂}}
end


# A mailbox in the Lauritzen-Spiegelhalter architecture.
mutable struct LSMailbox{T₁, T₂, T₃}
    factor::Union{Nothing, Factor{T₁, T₂}}
    cpd::Union{Nothing, CPD{T₃, T₂}}
end


# An architecture.
mutable struct Architecture{T₁, T₂, T₃, T₄}
    labels::Labels{T₁}
    factors::Vector{Factor{T₂, T₃}}
    tree::JoinTree
    assignments::Vector{Vector{Int}}
    mailboxes::Vector{T₄}
    messages_passed::Bool
end


# The Shenoy-Shafer architecture.
const SSArchitecture{T₁, T₂, T₃, T₄} = Architecture{T₁, T₂, T₃, SSMailbox{T₂, T₃, T₄}}


# The Lauritzen-Spiegelhalter architecture.
const LSArchitecture{T₁, T₂, T₃, T₄} = Architecture{T₁, T₂, T₃, LSMailbox{T₂, T₃, T₄}}


# Construct an empty mailbox for the Shenoy-Shafer architecture.
function SSMailbox{T₁, T₂, T₃}() where {T₁, T₂, T₃}
    SSMailbox{T₁, T₂, T₃}(nothing, nothing, nothing, nothing)
end


# Construct an empty mailbox for the Lauritzen-Spiegelhalter architecture.
function LSMailbox{T₁, T₂, T₃}() where {T₁, T₂, T₃}
    LSMailbox{T₁, T₂, T₃}(nothing, nothing)
end


# Construct a Shenoy-Shafer architecture with empty mailboxes.
function Architecture(
    labels::Labels,
    factors::Vector{Factor{T₁, T₂}},
    tree::JoinTree,
    assignments::Vector{Vector{Int}},
    atype::ShenoyShafer) where {T₁, T₂}

    T₃ = cpdtype(T₁)

    mailboxes = [SSMailbox{T₁, T₂, T₃}() for _ in labels]
    messages_passed = false

    Architecture(
        labels,
        factors,
        tree,
        assignments,
        mailboxes,
        messages_passed)
end


# Construct a Lauritzen-Spiegelhalter architecture with empty mailboxes.
function Architecture(
    labels::Labels,
    factors::Vector{Factor{T₁, T₂}},
    tree::JoinTree,
    assignments::Vector{Vector{Int}},
    atype::LauritzenSpiegelhalter) where {T₁, T₂}

    T₃ = cpdtype(T₁)

    mailboxes = [LSMailbox{T₁, T₂, T₃}() for _ in labels]
    messages_passed = false

    Architecture(
        labels,
        factors,
        tree,
        assignments,
        mailboxes,
        messages_passed)
end


# Construct an architecture.
function Architecture(
    model::GraphicalModel,
    elalg::EliminationAlgorithm,
    stype::SupernodeType,
    atype::ArchitectureType)

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

    Architecture(labels, factors, tree, assignments, atype)
end


# Answer a query.
function CommonSolve.solve(arch::SSArchitecture, query)
    @assert arch.messages_passed

    vars = [arch.labels.index[l] for l in query]

    for n in arch.tree.order
        node = IndexNode(arch.tree, n)
        sep, res = nodevalue(node)

        if vars ⊆ [sep; res]
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
function CommonSolve.solve!(arch::Architecture, query)
    if !arch.messages_passed
        pass_messages!(arch)
    end

    solve(arch, query)
end


# Answer a query.
function CommonSolve.solve(arch::LSArchitecture, query)
    @assert arch.messages_passed

    vars = [arch.labels.index[l] for l in query]

    for n in arch.tree.order
        node = IndexNode(arch.tree, n)
        sep, res = nodevalue(node)

        if vars ⊆ [sep; res]
            mbx = mailbox(arch, node.index)
            fac = mbx.factor

            fac = project(fac, vars)
            hom = permute(fac, vars)

            return hom
        end 
    end

    error("Query not covered by join tree.")
end


# Sample from an architecture.
function Base.rand(rng::AbstractRNG, arch::Architecture{<:Any, <:GaussianSystem})
    @assert arch.messages_passed

    x = Vector{Vector{Float64}}(undef, length(arch.labels))

    for n in reverse(arch.tree.order)
        node = IndexNode(arch.tree, n)

        mbx = mailbox(arch, node.index)
        rand!(rng, mbx.cpd, x)
    end

    Dict(zip(arch.labels, x))
end


function Base.rand(arch::Architecture{<:Any, <:GaussianSystem})
    rand(default_rng(), arch)
end


# Compute the mean of an architecture.
function Statistics.mean(arch::Architecture{<:Any, <:GaussianSystem})
    @assert arch.messages_passed

    x = Vector{Vector{Float64}}(undef, length(arch.labels))

    for n in reverse(arch.tree.order)
        node = IndexNode(arch.tree, n)

        mbx = mailbox(arch, node.index)
        mean!(mbx.cpd, x)
    end

    Dict(zip(arch.labels, x))
end


function pass_messages!(arch::SSArchitecture)
    for n in arch.tree.order          # Collect Phase:
        node = IndexNode(arch.tree, n)

        mbx = mailbox(arch, node.index)
        mbx.factor = factor(arch, node.index)
        mbx.message_to_parent, mbx.cpd = message_to_parent(arch, node.index)
    end

    for n in reverse(arch.tree.order) # Distribute Phase:
        node = IndexNode(arch.tree, n)

        mbx = mailbox(arch, node.index)
        mbx.message_from_parent = message_from_parent(arch, node.index)
    end

    arch.messages_passed = true
end


function pass_messages!(arch::LSArchitecture)
    for n in arch.tree.order
        node = IndexNode(arch.tree, n)

        mbx = mailbox(arch, node.index)
        mbx.factor = factor(arch, node.index)
    end

    for n in arch.tree.order         # Collect Phase:
        node = IndexNode(arch.tree, n)

        mbx = mailbox(arch, node.index)
        fac, mbx.cpd = message_to_parent(arch, node.index)

        if !isroot(node)
            mbx = mailbox(arch, parent(node).index)
            mbx.factor = combine(fac, mbx.factor)
        end
    end

    for n in reverse(arch.tree.order) # Distribute Phase:
        node = IndexNode(arch.tree, n)

        mbx = mailbox(arch, node.index)
        mbx.factor = combine(mbx.cpd, message_from_parent(arch, node.index))
    end

    arch.messages_passed = true
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


# Compute the message
# μ n → pa(n)
function message_to_parent(arch::SSArchitecture, n::Int)
    node = IndexNode(arch.tree, n)

    mbx = mailbox(arch, node.index)
    fac = mbx.factor

    for child in children(node)
        mbx = mailbox(arch, child.index)
        fac = combine(fac, mbx.message_to_parent)
    end

    disintegrate(fac, first(nodevalue(node)))
end


# Compute the message
# μ n → pa(n)
function message_to_parent(arch::LSArchitecture, n::Int)
    node = IndexNode(arch.tree, n)

    mbx = mailbox(arch, node.index)
    disintegrate(mbx.factor, first(nodevalue(node)))
end


# Compute the message
# μ pa(n) → n
function message_from_parent(arch::SSArchitecture{<:Any, T₁, T₂}, n::Int) where {T₁, T₂}
    node = IndexNode(arch.tree, n)

    if isroot(node)
        zero(Factor{T₁, T₂})
    else
        prnt = parent(node)
        mbx = mailbox(arch, prnt.index)
        fac = combine(mbx.factor, mbx.message_from_parent)

        for sibling in children(prnt)
            if node != sibling
                mbx = mailbox(arch, sibling.index)
                fac = combine(fac, mbx.message_to_parent)
            end
        end

        project(fac, first(nodevalue(node)))
    end
end


# Compute the message
# μ pa(n) → n
function message_from_parent(arch::LSArchitecture{<:Any, T₁, T₂}, n::Int) where {T₁, T₂}
    node = IndexNode(arch.tree, n)

    if isroot(node)
        zero(Factor{T₁, T₂})
    else
        mbx = mailbox(arch, parent(node).index)
        project(mbx.factor, first(nodevalue(node)))
    end
end

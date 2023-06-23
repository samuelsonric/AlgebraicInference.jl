struct KKT{T}
    cache::T
end

# Construct a KKT matrix of the form
# [ A B']
# [ B 0 ]
# where A is positive semidefinite.
function KKT(A::AbstractMatrix, B::AbstractMatrix, alg=KrylovJL_MINRES())
    A = convert(AbstractMatrix{Float64}, A)
    B = convert(AbstractMatrix{Float64}, B)
    m = size(A, 1)
    n = size(B, 1)
    K = [
        A B'
        B 0I(n) ]
    b = zeros(m + n)
    KKT(init(LinearProblem(K, b), alg))
end

# Solve for x:
# [ A B'] [ x ] = [ f ]
# [ B 0 ] [ y ]   [ g ]
# where A is positive semidefinite.
function solve!(K::KKT, f::AbstractVector, g::AbstractVector)
    n = length(f)
    f = convert(AbstractVector{Float64}, f)
    g = convert(AbstractVector{Float64}, g)
    K.cache.b = [f; g]
    solve!(K.cache)[1:n]
end

# Solve for X:
# [ A B'] [ X ] = [ F ]
# [ B 0 ] [ Y ]   [ G ]
# where A is positive semidefinite.
function solve!(K::KKT, F::AbstractMatrix, G::AbstractMatrix)
    n = size(F, 1)
    F = convert(AbstractMatrix{Float64}, F)
    G = convert(AbstractMatrix{Float64}, G)
    if size(F, 2) == 0
        F
    else
        mapslices([F; G]; dims=1) do b
            K.cache.b = b
            solve!(K.cache)[1:n]
        end
    end
end

# Compose Σ₁ and Σ₂ by sharing variables.
function combine(
    Σ₁::AbstractGaussianSystem{T},
    Σ₂::AbstractGaussianSystem{T},
    ls₁::Vector{Int},
    ls₂::Vector{Int},
    ix₁::Dict{Int, Int}) where T

    ls = copy(ls₁)
    ix = copy(ix₁)

    is = map(ls₂) do l₂
        get!(ix, l₂) do
            push!(ls, l₂)
            length(ls)
        end
    end
    
    n = length(ls)
    P = zeros(T, n, n)
    S = zeros(T, n, n)
    p = zeros(T, n)
    s = zeros(T, n)
    
    n₁ = length(ls₁)
    P[1:n₁, 1:n₁] = Σ₁.P
    S[1:n₁, 1:n₁] = Σ₁.S
    p[1:n₁] = Σ₁.p
    s[1:n₁] = Σ₁.s

    P[is, is] .+= Σ₂.P
    S[is, is] .+= Σ₂.S
    p[is] .+= Σ₂.p
    s[is] .+= Σ₂.s
  
    σ = Σ₁.σ + Σ₂.σ
    GaussianSystem(P, S, p, s, σ), ls, ix
end

function extend(
    Σ₁::AbstractGaussianSystem{T},
    ls₁::Vector{Int},
    ls₂::Vector{Int},
    ix₁::Dict{Int, Int}) where T

    ls = copy(ls₁)
    ix = copy(ix₁)

    for l₂ in ls₂
        get!(ix, l₂) do
            push!(ls, l₂)
            length(ls)
        end
    end

    n = length(ls)
    P = zeros(T, n, n)
    S = zeros(T, n, n)
    p = zeros(T, n)
    s = zeros(T, n)

    n₁ = length(ls₁)
    P[1:n₁, 1:n₁] = Σ₁.P
    S[1:n₁, 1:n₁] = Σ₁.S
    p[1:n₁] = Σ₁.p
    s[1:n₁] = Σ₁.s

    σ = Σ₁.σ
    GaussianSystem(P, S, p, s, σ), ls, ix
end

# Compute a variable elimination order using the minimum degree heuristic.
function mindegree!(pg::AbstractGraph)
    n = nv(pg)
    ls = collect(vertices(pg))
    order = zeros(Int, n)
    for i in 1:n
        v = argmin(map(v -> degree(pg, v), vertices(pg)))
        order[i] = ls[v]
        eliminate!(pg, ls, v)
    end
    order
end

# Compute a variable elimination order using the minimum fill heuristic.
function minfill!(pg::AbstractGraph)
    n = nv(pg)
    ls = collect(vertices(pg))
    fs = map(v -> fillins(pg, v), vertices(pg))
    order = zeros(Int, n)
    for i in 1:n
        v = argmin(fs)
        order[i] = ls[v]
        eliminate!(pg, ls, fs, v)
    end
    order
end

# The fill-in number of vertex v.
function fillins(pg::AbstractGraph, v::Integer)
    count = 0
    ns = neighbors(pg, v)
    n = length(ns)
    for i₁ in 1:n - 1
        for i₂ in i₁ + 1:n
            if !has_edge(pg, ns[i₁], ns[i₂])
                count += 1
            end
        end
    end
    count
end

# Eliminate the vertex v.
function eliminate!(pg::AbstractGraph, ls::Vector, v::Integer)
    ns = neighbors(pg, v)
    n = length(ns)
    for i₁ = 1:n - 1
        for i₂ = i₁ + 1:n
            add_edge!(pg, ns[i₁], ns[i₂])
        end
    end
    rem_vertex!(pg, v)
    ls[v] = ls[end]
    pop!(ls)
end

# Eliminate the vertex v.
# Adapted from https://github.com/JuliaQX/QXGraphDecompositions.jl/blob/
# 22ee3d75bcd267bf462eec8f03930af2129e34b7/src/LabeledGraph.jl#L326
function eliminate!(pg::AbstractGraph, ls::Vector, fs::Vector, v::Integer)
    ns = neighbors(pg, v)
    n = length(ns)
    for i₁ = 1:n - 1
        for i₂ = i₁ + 1:n
            if add_edge!(pg, ns[i₁], ns[i₂])
                ns₁ = neighbors(pg, ns[i₁])
                ns₂ = neighbors(pg, ns[i₂])
                for w in ns₁ ∩ ns₂
                    fs[w] -= 1
                end
                for w in ns₁
                    if w != ns[i₂] && !has_edge(pg, w, ns[i₂])
                        fs[ns[i₁]] += 1
                    end
                end
                for w in ns₂
                    if w != ns[i₁] && !has_edge(pg, w, ns[i₁])
                        fs[ns[i₂]] += 1
                    end
                end
            end
        end
    end
    for i₁ in 1:n
        ns₁ = neighbors(pg, ns[i₁])
        for w in ns₁
            if w != ns[i₁] && !has_edge(pg, w, ns[i₁])
                fs[w] -= 1
            end
        end
    end
    rem_vertex!(pg, v)
    ls[v] = ls[end]
    fs[v] = fs[end]
    pop!(ls)
    pop!(fs)
end

# Compute the message
# μ i -> pa(i)
function message_to_parent(node::JoinTree{T}) where T
    @assert !isroot(node)
    if isnothing(node.message_to_parent)
        factor = node.factor
        for child in node.children
            factor = combine(factor, message_to_parent(child)::Valuation{T})
        end
        project(factor, domain(factor) ∩ node.parent.domain)
    else
        node.message_to_parent
    end
end

# Compute the message
# μ pa(i) -> i
function message_from_parent(node::JoinTree{T}) where T
    @assert !isroot(node)
    if isnothing(node.message_from_parent)
        factor = node.parent.factor
        for sibling in node.parent.children
            if node.id != sibling.id
                factor = combine(factor, message_to_parent(sibling)::Valuation{T})
            end
        end
        if !isroot(node.parent)
            factor = combine(factor, message_from_parent(node.parent::JoinTree{T})::Valuation{T})
        end
        project(factor, domain(factor) ∩ node.domain)
    else
        node.message_from_parent
    end
end

# Compute the message
# μ i -> pa(i),
# caching intermediate computations.
function message_to_parent!(node::JoinTree{T}) where T
    @assert !isroot(node)
    if isnothing(node.message_to_parent)
        factor = node.factor
        for child in node.children
            factor = combine(factor, message_to_parent!(child)::Valuation{T})
        end
        node.message_to_parent = project(factor, domain(factor) ∩ node.parent.domain)
    end
    node.message_to_parent
end

# Compute the message
# μ pa(i) -> i,
# caching intermediate computations.
function message_from_parent!(node::JoinTree{T}) where T
    @assert !isroot(node)
    if isnothing(node.message_from_parent)
        factor = node.parent.factor
        for sibling in node.parent.children
            if node.id != sibling.id
                factor = combine(factor, message_to_parent!(sibling)::Valuation{T})
            end
        end
        if !isroot(node.parent)
            factor = combine(factor, message_from_parent!(node.parent::JoinTree{T})::Valuation{T})
        end
        node.message_from_parent = project(factor, domain(factor) ∩ node.domain)
    end
    node.message_from_parent
end

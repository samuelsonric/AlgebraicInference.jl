struct KKT{T₁, T₂, T₃, T₄, T₅}
    A::T₁
    B::T₂
    U::T₃
    cache₁::T₄
    cache₂::T₅
end

# Construct a KKT matrix of the form
# [ A B']
# [ B 0 ]
# where A is positive semidefinite.
function KKT(A, B, alg=KrylovJL_MINRES(); atol=1e-8)
    U = nullspace(B; atol)'
    A₁ = B * B'
    A₂ = U * A * U'
    b₁ = zeros(size(B, 1))
    b₂ = zeros(size(U, 1))
    cache₁ = init(LinearProblem(A₁, b₁), alg)
    cache₂ = init(LinearProblem(A₂, b₂), alg)
    KKT(A, B, U, cache₁, cache₂)
end

# Solve for x:
# [ A B'] [ x ] = [ f ]
# [ B 0 ] [ y ]   [ g ]
# where A is positive semidefinite.
function solve!(K::KKT, f::AbstractVector, g::AbstractVector)
    K.cache₁.b = g
    x₁ = K.B' * solve!(K.cache₁)
    K.cache₂.b = K.U * (f - K.A * x₁)
    x₂ = K.U' * solve!(K.cache₂)
    x₁ + x₂
end

function solve!(K::KKT, f::AbstractVector, g::ZerosVector)
    K.cache₂.b = K.U * f
    K.U' * solve!(K.cache₂)
end

# Solve for X:
# [ A B'] [ X ] = [ F ]
# [ B 0 ] [ Y ]   [ G ]
# where A is positive semidefinite.
function solve!(K::KKT, F::AbstractMatrix, G::AbstractMatrix)
    m, n = size(F)
    X = zeros(m, n)
    for i in 1:n
        X[:, i] = solve!(K, F[:, i], G[:, i])
    end
    X
end

# Compute a variable elimination order using the minimum degree heuristic.
function mindegree!(graph::AbstractGraph)
    n = nv(graph)
    ls = collect(vertices(graph))
    order = zeros(Int, n)
    for i in 1:n
        v = argmin(map(v -> degree(graph, v), vertices(graph)))
        order[i] = ls[v]
        eliminate!(graph, ls, v)
    end
    order
end

# Compute a variable elimination order using the minimum fill heuristic.
function minfill!(graph::AbstractGraph)
    n = nv(graph)
    ls = collect(vertices(graph))
    fs = map(v -> fillins(graph, v), vertices(graph))
    order = zeros(Int, n)
    for i in 1:n
        v = argmin(fs)
        order[i] = ls[v]
        eliminate!(graph, ls, fs, v)
    end
    order
end

# The fill-in number of vertex v.
function fillins(graph::AbstractGraph, v::Integer)
    count = 0
    ns = neighbors(graph, v)
    n = length(ns)
    for i₁ in 1:n - 1
        for i₂ in i₁ + 1:n
            if !has_edge(graph, ns[i₁], ns[i₂])
                count += 1
            end
        end
    end
    count
end

# Eliminate the vertex v.
function eliminate!(graph::AbstractGraph, ls::Vector, v::Integer)
    ns = neighbors(graph, v)
    n = length(ns)
    for i₁ = 1:n - 1
        for i₂ = i₁ + 1:n
            add_edge!(graph, ns[i₁], ns[i₂])
        end
    end
    rem_vertex!(graph, v)
    ls[v] = ls[end]
    pop!(ls)
end

# Eliminate the vertex v.
# Adapted from https://github.com/JuliaQX/QXGraphDecompositions.jl/blob/
# 22ee3d75bcd267bf462eec8f03930af2129e34b7/src/LabeledGraph.jl#L326
function eliminate!(graph::AbstractGraph, ls::Vector, fs::Vector, v::Integer)
    ns = neighbors(graph, v)
    n = length(ns)
    for i₁ = 1:n - 1
        for i₂ = i₁ + 1:n
            if add_edge!(graph, ns[i₁], ns[i₂])
                ns₁ = neighbors(graph, ns[i₁])
                ns₂ = neighbors(graph, ns[i₂])
                for w in ns₁ ∩ ns₂
                    fs[w] -= 1
                end
                for w in ns₁
                    if w != ns[i₂] && !has_edge(graph, w, ns[i₂])
                        fs[ns[i₁]] += 1
                    end
                end
                for w in ns₂
                    if w != ns[i₁] && !has_edge(graph, w, ns[i₁])
                        fs[ns[i₂]] += 1
                    end
                end
            end
        end
    end
    for i₁ in 1:n
        ns₁ = neighbors(graph, ns[i₁])
        for w in ns₁
            if w != ns[i₁] && !has_edge(graph, w, ns[i₁])
                fs[w] -= 1
            end
        end
    end
    rem_vertex!(graph, v)
    ls[v] = ls[end]
    fs[v] = fs[end]
    pop!(ls)
    pop!(fs)
end

# Compute the message
# μ i -> pa(i)
function message_to_parent(node::JoinTree{T}, objects) where T
    @assert !isroot(node)
    if isnothing(node.message_to_parent)
        factor = node.factor
        for child in node.children
            message = message_to_parent(child, objects)::Valuation{T}
            factor = combine(factor, message, objects)
        end
        project(factor, domain(factor) ∩ node.parent.domain, objects)
    else
        node.message_to_parent
    end
end

# Compute the message
# μ pa(i) -> i
function message_from_parent(node::JoinTree{T}, objects) where T
    @assert !isroot(node)
    if isnothing(node.message_from_parent)
        factor = node.parent.factor
        for sibling in node.parent.children
            if node.id != sibling.id
                message = message_to_parent(sibling, objects)::Valuation{T}
                factor = combine(factor, message, objects)
            end
        end
        if !isroot(node.parent)
            message = message_from_parent(node.parent::JoinTree{T}, objects)::Valuation{T}
            factor = combine(factor, message, objects)
        end
        project(factor, domain(factor) ∩ node.domain, objects)
    else
        node.message_from_parent
    end
end

# Compute the message
# μ i -> pa(i),
# caching intermediate computations.
function message_to_parent!(node::JoinTree{T}, objects) where T
    @assert !isroot(node)
    if isnothing(node.message_to_parent)
        factor = node.factor
        for child in node.children
            message = message_to_parent!(child, objects)::Valuation{T}
            factor = combine(factor, message, objects)
        end
        node.message_to_parent = project(factor, domain(factor) ∩ node.parent.domain, objects)
    end
    node.message_to_parent
end

# Compute the message
# μ pa(i) -> i,
# caching intermediate computations.
function message_from_parent!(node::JoinTree{T}, objects) where T
    @assert !isroot(node)
    if isnothing(node.message_from_parent)
        factor = node.parent.factor
        for sibling in node.parent.children
            if node.id != sibling.id
                message = message_to_parent!(sibling, objects)::Valuation{T}
                factor = combine(factor, message, objects)
            end
        end
        if !isroot(node.parent)
            message = message_from_parent!(node.parent::JoinTree{T}, objects)::Valuation{T}
            factor = combine(factor, message, objects)
        end
        node.message_from_parent = project(factor, domain(factor) ∩ node.domain, objects)
    end
    node.message_from_parent
end

struct KKT{T}
    cache::T
end

# Construct a KKT matrix of the form
# [ A B']
# [ B 0 ]
# where A is positive semidefinite.
function KKT(A::AbstractMatrix, B::AbstractMatrix)
    A = convert(AbstractMatrix{Float64}, A)
    B = convert(AbstractMatrix{Float64}, B)
    n = size(B, 1)
    K = [
        A B'
        B Matrix(0I, n, n) ]
    b = zeros(size(K, 1))
    KKT(init(LinearProblem(K, b), KrylovJL_MINRES()))
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

# Compute the vacuous extension
# Σ ↑ l
function extend(Σ::GaussianSystem{
    <:AbstractMatrix{T₁},
    <:AbstractMatrix{T₂},
    <:AbstractVector{T₃},
    <:AbstractVector{T₄}}, _l, l) where {T₁, T₂, T₃, T₄}

    n = length(l); _n = length(_l)
    P = zeros(T₁, n, n)
    S = zeros(T₂, n, n)
    p = zeros(T₃, n)
    s = zeros(T₄, n)
    
    for _i in 1:_n
        i = findfirst(X -> X == _l[_i], l)
        P[i, i] = Σ.P[_i, _i]
        S[i, i] = Σ.S[_i, _i]        
        p[i] = Σ.p[_i]
        s[i] = Σ.s[_i]
        
        for _j in _i+1:_n
            j = findfirst(X -> X == _l[_j], l)
            P[i, j] = Σ.P[_i, _j]
            S[i, j] = Σ.S[_i, _j]
            P[j, i] = Σ.P[_j, _i]
            S[j, i] = Σ.S[_j, _i]
        end
    end
   
    σ = Σ.σ 
    GaussianSystem(P, S, p, s, σ)
end

# Construct the primal graph of the knowledge base kb.
function primalgraph(kb::Vector{<:Valuation{T}}) where T
    pg = Graph()
    ls = T[]
    l2v = Dict{T, Int}()
    for dom in map(domain, kb)
        n = length(dom)
        for i in 1:n
            if !haskey(l2v, dom[i])
                l2v[dom[i]] = nv(pg) + 1
                add_vertex!(pg)
                push!(ls, dom[i])
            end
            for j in 1:i - 1
                add_edge!(pg, l2v[dom[i]], l2v[dom[j]]) 
            end
        end
    end
    pg, ls
end

# Compute a variable elimination order using the min-width heuristic.
function minwidth!(pg::AbstractGraph, ls::Vector{T}, query) where T
    n = nv(pg) - length(query)
    order = Vector{T}(undef, n)
    for i in 1:n
        v = map(vertices(pg)) do v
            if ls[v] in query
                typemax(Int)    
            else
                degree(pg, v)
            end
        end |> argmin
        order[i] = ls[v]
        eliminate!(pg, ls, v)
    end
    order
end

# Compute a variable elimination order using the min-fill heuristic.
function minfill!(pg::AbstractGraph, ls::Vector{T}, query) where T
    n = nv(pg) - length(query)
    order = Vector{T}(undef, n)
    for i in 1:n
        v = map(vertices(pg)) do v
            if ls[v] in query
                typemax(Int)    
            else
                 fill_in_number(pg, v)
            end
        end |> argmin
        order[i] = ls[v]
        eliminate!(pg, ls, v)
    end
    order
end

# The fill-in number of vertex v.
function fill_in_number(pg::AbstractGraph, v::Integer)
    fi = 0
    ns = neighbors(pg, v)
    n = length(ns)
    for i in 1:n - 1
        for j in i + 1:n
            if !has_edge(pg, ns[i], ns[j])
                fi += 1
            end
        end
    end
    fi
end

# Eliminate the vertex v.
function eliminate!(pg::AbstractGraph, ls::Vector, v::Integer)
    ns = neighbors(pg, v)
    n = length(ns)
    for i = 1:n - 1
        for j = i + 1:n
            add_edge!(pg, ns[i], ns[j])
        end
    end
    rem_vertex!(pg, v)
    ls[v] = ls[end]
    pop!(ls)
end

# Compute the message
# μ i -> pa(i)
function message_to_parent(node::JoinTree{T}) where T
    @assert !isroot(node)
    if isnothing(node.message_to_parent)
        factor = node.factor
        for child in node.children
            factor = combine(factor, message_to_parent(child)::T)
        end
        project(factor, domain(factor) ∩ node.parent.domain)
    else
        node.message_to_parent
    end
end

# Compute the message
# μ pa(i) -> i
function message_from_parent(node::N) where {T, N <: JoinTree{T}}
    @assert !isroot(node)
    if isnothing(node.message_from_parent)
        factor = node.factor
        for sibling in node.parent.children
            if node.id != sibling.id
                factor = combine(factor, message_to_parent(sibling)::T)
            end
        end
        if !isroot(node.parent)
            factor = combine(factor, message_from_parent(node.parent::N)::T)
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
            factor = combine(factor, message_to_parent!(child)::T)
        end
        node.message_to_parent = project(factor, domain(factor) ∩ node.parent.domain)
    end
    node.message_to_parent
end

# Compute the message
# μ pa(i) -> i,
# caching intermediate computations.
function message_from_parent!(node::N) where {T, N <: JoinTree{T}}
    @assert !isroot(node)
    if isnothing(node.message_from_parent)
        factor = node.factor
        for sibling in node.parent.children
            if node.id != sibling.id
                factor = combine(factor, message_to_parent!(sibling)::T)
            end
        end
        if !isroot(node.parent)
            factor = combine(factor, message_from_parent!(node.parent::N)::T)
        end
        node.message_from_parent = project(factor, domain(factor) ∩ node.domain)
    end
    node.message_from_parent
end

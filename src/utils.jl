# Solve for x:
# [ A B'] [ x ] = [ f ]
# [ B 0 ] [ y ]   [ g ]
# where A is positive semidefinite.
function saddle(A::AbstractMatrix, B::AbstractMatrix, f::AbstractVector, g::AbstractVector)
    n = size(A, 1)
    L = [
        A B'
        B 0I
    ]
    R = [f; g]
    (qr(L, ColumnNorm()) \ R)[1:n]
end

# Solve for X:
# [ A B'] [ X ] = [ F ]
# [ B 0 ] [ Y ]   [ G ]
# where A is positive semidefinite.
function saddle(A::AbstractMatrix, B::AbstractMatrix, F::AbstractMatrix, G::AbstractMatrix)
    n = size(A, 1)
    L = [
        A B'
        B 0I
    ]
    R = [F; G]
    (qr(L, ColumnNorm()) \ R)[1:n, :]
end

# Compute the pushforward
# M*Σ
# where M is surjective.
function pushfwd_epi(M::AbstractMatrix, Σ::GaussianSystem)
    @assert size(M, 2) == length(Σ)
    P, S = Σ.P, Σ.S
    p, s = Σ.p, Σ.s
    σ = Σ.σ

    m, n = size(M)

    A = saddle(P, [S; M], zeros(n, m), [zeros(n, m); I(m)])
    a = saddle(P, [S; M], p, [s; zeros(m)])

    GaussianSystem(
        A' * P * A,
        A' * S * A,
        A' * (p - P * a),
        A' * (s - S * a),
        a' * (s - S * a) * -1 + σ - s' * a)
end

# Compute the pushforward
# M*Σ
function pushfwd(M::AbstractMatrix, Σ::GaussianSystem)
    Σ = pushfwd_epi(M, Σ)
    V = nullspace(M')
    GaussianSystem(Σ.P, Σ.S + V * V', Σ.p, Σ.s, Σ.σ)
end

# Compute the marginal of Σ along the indices in m.
function marginal(Σ::GaussianSystem, m::AbstractVector{Bool})
    P, S = Σ.P, Σ.S
    p, s = Σ.p, Σ.s
    σ = Σ.σ

    n = .!m

    A = saddle(P[n, n], S[n, n], P[n, m], S[n, m])
    a = saddle(P[n, n], S[n, n], p[n], s[n])

    GaussianSystem(
        P[m, m] + A' * P[n, n] * A - P[m, n] * A - A' * P[n, m],
        S[m, m] + A' * S[n, n] * A - S[m, n] * A - A' * S[n, m],
        p[m]    + A' * P[n, n] * a - P[m, n] * a - A' * p[n],
        s[m]    + A' * S[n, n] * a - S[m, n] * a - A' * s[n],
        σ       + a' * S[n, n] * a - s[n]'   * a - a' * s[n])
end


# Compute the vacuous extension
# Σ ↑ l
function extend(l, _l, Σ::GaussianSystem{
    <:AbstractMatrix{T₁},
    <:AbstractMatrix{T₂},
    <:AbstractVector{T₃},
    <:AbstractVector{T₄},
    <:Any}) where {T₁, T₂, T₃, T₄}
    
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

# Compute the message
# μ i -> pa(i)
function message_to_parent(node::Architecture)
    @assert !isroot(node)
    if isnothing(node.message_to_parent)
        factor = node.factor
        for child in node.children
            factor = combine(factor, message_to_parent(child))
        end
        project(factor, domain(factor) ∩ node.parent.domain)
    else
        node.message_to_parent
    end
end

# Compute the message
# μ pa(i) -> i
function message_from_parent(node::Architecture)
    @assert !isroot(node)
    if isnothing(node.message_from_parent)
        factor = node.factor
        for sibling in node.parent.children
            if node.id != sibling.id
                factor = combine(factor, message_to_parent(sibling))
            end
        end
        if !isroot(node.parent)
            factor = combine(factor, message_from_parent(node.parent))
        end
        project(factor, domain(factor) ∩ node.domain)
    else
        node.message_from_parent
    end
end

# Compute the message
# μ i -> pa(i),
# caching intermediate computations.
function message_to_parent!(node::Architecture)
    @assert !isroot(node)
    if isnothing(node.message_to_parent)
        factor = node.factor
        for child in node.children
            factor = combine(factor, message_to_parent!(child))
        end
        node.message_to_parent = project(factor, domain(factor) ∩ node.parent.domain)
    end
    node.message_to_parent
end

# Compute the message
# μ pa(i) -> i,
# caching intermediate computations.
function message_from_parent!(node::Architecture)
    @assert !isroot(node)
    if isnothing(node.message_from_parent)
        factor = node.factor
        for sibling in node.parent.children
            if node.id != sibling.id
                factor = combine(factor, message_to_parent!(sibling))
            end
        end
        if !isroot(node.parent)
            factor = combine(factor, message_from_parent!(node.parent))
        end
        node.message_from_parent = project(factor, domain(factor) ∩ node.domain)
    end
    node.message_from_parent
end

# The fill-in number of vertex v.
function fill_in_number(g::MetaGraph, v)
    fi = 0
    ns = neighbors(g, v)
    len = length(ns)
    for i in 1:len-1
        for j in i+1:len
            if !has_edge(g, ns[i], ns[j])
                fi += 1
            end
        end
    end
    fi
end

# Eliminate the vertex v.
function eliminate!(g::MetaGraph, v)
    ns = neighbors(g, v)
    len = length(ns)
    for i in 1:len-1
        for j in i+1:len
            add_edge!(g, label_for(g, ns[i]), label_for(g, ns[j]))
        end
    end
    rem_vertex!(g, v)
end

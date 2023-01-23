"""
    QuadraticFunction(Q, a)

A convex quadratic function ``f`` of the form
```math
f(x) = \\langle x, Qx + a \\rangle,
```
where ``Q`` is positive semidefinite.
"""
struct QuadraticFunction{T₁ <: AbstractMatrix, T₂ <: AbstractVector}
    Q::T₁
    a::T₂
end

"""
    QuadraticFunction(Q::UniformScaling, a::AbstractVector)

Construct the convex quadratic function ``f(x) = \\langle x, Qx + a \\rangle``, where ``Q`` is positive semidefinite.
"""
function QuadraticFunction(Q::UniformScaling, a::AbstractVector)
    Q = Q(length(a))
    QuadraticFunction(Q, a)
end

"""
    QuadraticFunction(a::AbstractVector)

Construct the linear function ``f(x) = \\langle x, a \\rangle``.
"""
function QuadraticFunction(a::AbstractVector)
    Q = 0I
    QuadraticFunction(Q, a)
end

"""
    QuadraticFunction(Q::AbstractMatrix)

Construct the convex quadratic function ``f(x) = \\langle x, Qx \\rangle``, where ``Q`` is positive semidefinite.
"""
function QuadraticFunction(Q::AbstractMatrix)
    a = zeros(size(Q, 1))
    QuadraticFunction(Q, a)
end

length(f::QuadraticFunction) = length(f.a)

"""
    *(f::QuadraticFunction, M::Union{UniformScaling, AbstractMatrix})

Compute the inverse image of ``f`` under ``M``, given by
```math
(fM)(y) = f(My).
```
"""
function *(f::QuadraticFunction, M::Union{UniformScaling, AbstractMatrix})
    Q = M' * f.Q * M
    a = M' * f.a
    QuadraticFunction(Q, a)
end

# Computes the *closure of* the image of f under M.
"""
    *(M::Union{UniformScaling, AbstractMatrix}, f::QuadraticFunction)

Compute the image of ``f`` under ``M``, given by
```math
(Mf)(y) = \\inf \\{ f(x) \\mid y = Mx \\}.
```
Returns a quintuple ``(Q, a, \\alpha, B, b).`` If ``b \\neq 0``, then ``Mf = -\\infty``. Otherwise,
```math
(Mf)(y) = \\begin{cases}
    \\langle y, \\frac{1}{2}Qy + a \\rangle + \\alpha    & By = 0 \\\\
    \\infty                                     & \\text{else}
\\end{cases}.
```
"""
function *(M::Union{UniformScaling, AbstractMatrix}, f::QuadraticFunction)
    M⁺ = pinv(M)
    Q⁺ = pinv((I - M⁺ * M) * f.Q * (I - M⁺ * M))

    Q = M⁺' * (I - f.Q * Q⁺) * f.Q * M⁺
    a = M⁺' * (I - f.Q * Q⁺) * f.a
    α = -1/2 * f.a' * Q⁺ * f.a
    B = I - M * M⁺
    b = (I - M⁺ * M) * (I - f.Q * Q⁺) * f.a
    (Q, a, α, B, b)
end

"""
    oplus(f₁::QuadraticFunction, f₂::QuadraticFunction)

Compute the direct sum of ``f_1`` and ``f_2``, given by 
```math
(f_1 \\oplus f_2)(x, y) = f_1(x) + f_2(y).
```
"""
function oplus(f₁::QuadraticFunction, f₂::QuadraticFunction)
    Q = f₁.Q ⊕ f₂.Q
    a = [f₁.a; f₂.a]
    QuadraticFunction(Q, a)
end

⊕(f₁::QuadraticFunction, f₂::QuadraticFunction) = oplus(f₁, f₂)

"""
    QuadraticBifunction(L, R, f)

A concave quadratic bifunction ``F`` of the form
```math
F(x, y) = -f^*(Lx - Ry),
```
where ``f^*`` is the convex conjuate of ``f``.
"""
struct QuadraticBifunction{T₁ <: AbstractMatrix, T₂ <: AbstractMatrix, T₃, T₄}
    L::T₁
    R::T₂
    f::QuadraticFunction{T₃, T₄}
end

"""
    QuadraticBifunction(L::AbstractMatrix)

Construct the concave indicator bifunction corresponding to ``L``, given by
```math
F(x, y) = \\begin{cases}
0           & Lx = y \\\\
-\\infty    & \\text{else}
\\end{cases}.
```
"""
function QuadraticBifunction(L::AbstractMatrix)
    n = size(L, 1)
    R = Matrix(I, n, n)
    f = QuadraticFunction(zeros(n))
    QuadraticBifunction(L, R, f)
end

"""
    QuadraticBifunction(f::QuadraticFunction)

Construct the concave bifunction ``F(*, y) = -f^*(-y)``, where ``f^*`` is the convex conjugate of ``f``.
"""
function QuadraticBifunction(f::QuadraticFunction)
    n = length(f)
    L = zeros(n, 0)
    R = Matrix(I, n, n)
    QuadraticBifunction(L, R, f)
end

"""
    adjoint(F::QuadraticBifunction)

Compute the adjoint of ``F``, given by
```math
F^*(y^*, x^*) = \\sup \\{ F(x, y) - \\langle y, y^* \\rangle + \\langle x, x^* \\rangle \\mid x, y \\}.
```
Returns a quintuple ``(Q, a, \\alpha, B, b).`` If ``b \\neq 0``, then ``F^* = -\\infty``. Otherwise,
```math
(F^*)(y, x) = \\begin{cases}
    \\langle (y, x), \\frac{1}{2}Q(y, x) + a \\rangle + \\alpha  & By = 0 \\\\
    \\infty                                             & \\text{else}
\\end{cases}.
```
"""
adjoint(F::QuadraticBifunction) = [F.R F.L]' * F.f

"""
    QuadDom(n)

The Euclidian space ``\\mathbb{R}^n``.
"""
struct QuadDom{T <: Integer}
    n::T
end

function pushout(L::AbstractMatrix, R::AbstractMatrix)
    n = size(L, 1)
    K = nullspace([L' -R'])
    Lₚ = K[1:n, :]'
    Rₚ = K[n+1:end, :]'
    (Lₚ, Rₚ)
end

@instance ThAbelianBicategoryRelations{QuadDom, QuadraticBifunction} begin
    
    """
        dom(F::QuadraticBifunction)

    The domain of a bifunction is the dimension of its first variable.
    """
    dom(F::QuadraticBifunction) = QuadDom(size(F.L, 2))

    """
        codom(F::QuadraticBifunction)

    The codomain of a bifunction is the dimension of its second variable.
    """
    codom(F::QuadraticBifunction) = QuadDom(size(F.R, 2))
 
    """
        mzero(::Type{QuadDom})

    The Euclidean space ``\\mathbb{R}^0``.
    """
    mzero(::Type{QuadDom}) = QuadDom(0)

    """
        oplus(X::QuadDom, Y::QuadDom)

    Compute the direct sum of ``X`` and ``Y``.
    """
    oplus(X::QuadDom, Y::QuadDom) = QuadDom(X.n + Y.n)
   
    """
        dagger(F::QuadraticBifunction)

    Compute the dagger of ``F``, given by
    ```math
    F^\\dagger(y, x) = F(x, y).
    ```
    """
    dagger(F::QuadraticBifunction) = QuadraticBifunction(F.R, F.L, F.f * -I)

    """
        compose(F₁::QuadraticBifunction, F₂::QuadraticFunction)
    
    Compute the product of ``F_1`` and ``F_2``, given by
    ```math
        (F_2F_1)(x, z) = \\sup \\{ F_1(x, y) + F_2(y, z) \\mid y \\}.
    ```
    """
    function compose(F₁::QuadraticBifunction, F₂::QuadraticBifunction)
        Lₚ, Rₚ = pushout(F₁.R, F₂.L)
        L = Lₚ * F₁.L
        R = Rₚ * F₂.R
        f = (F₁.f ⊕ F₂.f) * [Lₚ Rₚ]'
        QuadraticBifunction(L, R, f)
    end

    """
        oplus(F₁::QuadraticBifunction, F₂::QuadraticBifunction)

    Compute the direct sum of ``F_1`` and ``F_2``, given by
    ```math
        (F_1 \\oplus F_2)((x_1, x_2), (y_1, y_2)) = F_1(x_1, y_1) + F_2(x_2, y_2).
    ```
    """
    function oplus(F₁::QuadraticBifunction, F₂::QuadraticBifunction)
        L = F₁.L ⊕ F₂.L
        R = F₁.R ⊕ F₂.R
        f = F₁.f ⊕ F₂.f
        QuadraticBifunction(L, R, f)
    end

    """
        meet(F₁::QuadraticBifunction, F₂::QuadraticBifunction)

    Compute the sum of ``F_1`` and ``F_2``, given by
    ```math
        (F_1 + F_2)(x, y) = F_1(x, y) + F_2(x, y).
    ```
    """
    meet(F₁::QuadraticBifunction, F₂::QuadraticBifunction) = Δ(X) ⋅ (F₁ ⊕ F₂) ⋅ ∇(X)

    """
        join(F₁::QuadraticBifunction, F₂::QuadraticBifunction)
    
    Compute the supremal convolution of ``F_1``  and ``F_2``, given by
    ```math
        (F_1 \\Box F_2)(x, y) = \\sup \\{ F_1(x_1, y_1) + F_2(x - x_1, y - y_1) \\mid x, y \\}.
    ```
    """
    join(F₁::QuadraticBifunction, F₂::QuadraticBifunction) = coplus(X) ⋅ (F₁ ⊕ F₂) ⋅ plus(X)

    """
        id(X::QuadDom)

    Construct the concave indicator bifunction
    ```math
    F(x, y) = \\begin{cases}
        0           & x = y \\\\
        -\\infty    & \\text{else}
    \\end{cases}.
    ```
    """
    function id(X::QuadDom)
        L = Matrix(I, X.n, X.n)
        QuadraticBifunction(L)
    end

    """
        zero(X::QuadDom)

    Construct the constant bifunction ``F(*, y) = 0.``
    """
    function zero(X::QuadDom)
        L = zeros(X.n, 0)
        QuadraticBifunction(L)
    end

    """
        delete(X::QuadDom)

    Construct the constant bifunction ``F(x, *) = 0.``
    """
    function delete(X::QuadDom)
        L = zeros(0, X.n)
        QuadraticBifunction(L)
    end
    
    """
        mcopy(X::QuadDom)

    Construct the concave indicator bifunction
    ```math
    F(x, (y_1, y_2)) = \\begin{cases}
        0           & x = y_1 = y_2 \\\\
        -\\infty    & \\text{else}
    \\end{cases}.
    ```
    """
    function mcopy(X::QuadDom)
        L = [Matrix(I, X.n, X.n); Matrix(I, X.n, X.n)]
        QuadraticBifunction(L)
    end

    """
        plus(X::QuadDom)

    Construct the concave indicator bifunction
    ```math
    F((x_1, x_2), y) = \\begin{cases}
        0           & x_1 + x_2 = y \\\\
        -\\infty    & \\text{else}
    \\end{cases}.
    ```
    """
    function plus(X::QuadDom)
        L = [Matrix(I, X.n, X.n) Matrix(I, X.n, X.n)]
        QuadraticBifunction(L)
    end
    
    """
        dunit(X::QuadDom)

    Construct the concave indicator bifunction
    ```math
    F(*, (y_1, y_2)) = \\begin{cases}
        0           & y_1 = y_2 \\\\
        -\\infty    & \\text{else}
    \\end{cases}.
    ```
    """
    dunit(X::QuadDom) = □(X) ⋅ Δ(X)
    
    """
        cozero(X::QuadDom)
    
    Construct the concave indicator bifunction
    ```math
    F(x, *) = \\begin{cases}
        0           & x = 0 \\\\
        -\\infty    & \\text{else}
    \\end{cases}.
    ```
    """
    cozero(X::QuadDom) = dagger(zero(X))

    """
        create(X::QuadDom)

    Construct the concave indicator bifunction
    ```math
    F(*, y) = \\begin{cases}
        0           & y = 0 \\\\
        -\\infty    & \\text{else}
    \\end{cases}
    ```
    """
    create(X::QuadDom) = dagger(delete(X))
    
    """
        mmerge(X::QuadDom)

    Construct the concave indicator bifunction
    ```math
    F((x_1, x_2), y) = \\begin{cases}
        0           & x_1 = x_2 = y \\\\
        -\\infty    & \\text{else}
    \\end{cases}.
    ```
    """
    mmerge(X::QuadDom) = dagger(mcopy(X))

    """
        coplus(X::QuadDom)

    Construct the concave indicator bifunction
    ```math
    F((x_1, x_2), y) = \\begin{cases}
        0           & x_1 + x_2 = y \\\\
        -\\infty    & \\text{else}
    \\end{cases}
    ```
    """
    coplus(X::QuadDom) = dagger(plus(X))

    """
        dcounit(X::QuadDom)

    Construct the concave indicator bifunction
    ```math
    F((x_1, x_2), *) = \\begin{cases}
        0           & x_1 = x_2 \\\\
        -\\infty    & \\text{else}
    \\end{cases}.
    ```
    """
    dcounit(X::QuadDom) = dagger(dunit(X)) 

    """
        swap(X::QuadDom, Y::QuadDom)
    
    Construct the concave indicator bifunction
    ```math
    F((x_1, x_2), (y_1, y_2)) = \\begin{cases}
        0           & x_1 = y_2, x_2 = y_1 \\\\
        -\\infty    & \\text{else}
    \\end{cases}.
    ```
    """
    function swap(X::QuadDom, Y::QuadDom)
        L = [zeros(Y.n, X.n) Matrix(I, Y.n, Y.n); Matrix(I, X.n, X.n) zeros(X.n, Y.n)]
        QuadraticBifunction(L)
    end

    """
        top(X::QuadDom, Y::QuadDom)

    Construct the concave indicator bifunction
    ```math
    F(x, y) = \\begin{cases}
        0           & x = y = 0 \\\\
        -\\infty    & \\text{else}
    \\end{cases}.
    ```
    """
    top(X::QuadDom, Y::QuadDom) = ◊(X) ⋅ □(Y)

    """
        bottom(X::QuadDom, Y::QuadDom)

    Construct the constant bifunction ``F(x, y) = 0.``
    """
    bottom(X::QuadDom, Y::QuadDom) = cozero(X) ⋅ zero(Y)
end

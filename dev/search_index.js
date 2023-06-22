var documenterSearchIndex = {"docs":
[{"location":"api/#Library-Reference","page":"Library Reference","title":"Library Reference","text":"","category":"section"},{"location":"api/#Systems","page":"Library Reference","title":"Systems","text":"","category":"section"},{"location":"api/","page":"Library Reference","title":"Library Reference","text":"GaussianSystem\nGaussianSystem(::AbstractMatrix, ::AbstractMatrix, ::AbstractVector, ::AbstractVector, ::Any)\n\nnormal\nkernel\n\nlength(::GaussianSystem)\ncov(::GaussianSystem)\ninvcov(::GaussianSystem)\nvar(::GaussianSystem)\nmean(::GaussianSystem)\n\noapply(::AbstractUWD, ::AbstractVector{<:GaussianSystem})","category":"page"},{"location":"api/#AlgebraicInference.GaussianSystem","page":"Library Reference","title":"AlgebraicInference.GaussianSystem","text":"GaussianSystem{T₁, T₂, T₃, T₄, T₅}\n\nA Gaussian system.\n\n\n\n\n\n","category":"type"},{"location":"api/#AlgebraicInference.GaussianSystem-Tuple{AbstractMatrix, AbstractMatrix, AbstractVector, AbstractVector, Any}","page":"Library Reference","title":"AlgebraicInference.GaussianSystem","text":"GaussianSystem{T₁, T₂, T₃, T₄, T₅}(P, S, p, s, σ) where {\n    T₁ <: AbstractMatrix,\n    T₂ <: AbstractMatrix,\n    T₃ <: AbstractVector,\n    T₄ <: AbstractVector,\n    T₅ <: Real}\n\nConstruct a Gaussian system by specifying its energy function. \n\nYou should set σ equal to s^mathsfT S^+ s, where S^+ is the Moore-Penrose psuedoinverse of S.\n\n\n\n\n\n","category":"method"},{"location":"api/#AlgebraicInference.normal","page":"Library Reference","title":"AlgebraicInference.normal","text":"normal(μ::AbstractVector, Σ::AbstractMatrix)\n\nConstruct a multivariate normal distribution with mean vector μ and covariance matrix Σ.\n\n\n\n\n\nnormal(μ::Real, σ::Real)\n\nConstruct a normal distribution with mean μ and standard deviation σ.\n\n\n\n\n\n","category":"function"},{"location":"api/#AlgebraicInference.kernel","page":"Library Reference","title":"AlgebraicInference.kernel","text":"kernel(L::AbstractMatrix, μ::AbstractVector, Σ::AbstractMatrix)\n\nConstruct a conditional distribution of the form (y mid x) sim mathcalN(Lx + mu Sigma)\n\n\n\n\n\nkernel(l::AbstractVector, μ::Real, σ::Real)\n\nConstruct a conditional distribution of the form (y mid x) sim mathcalN(l^mathsfTx + mu sigma^2)\n\n\n\n\n\n","category":"function"},{"location":"api/#Base.length-Tuple{GaussianSystem}","page":"Library Reference","title":"Base.length","text":"length(Σ::GaussianSystem)\n\nGet the dimension of Σ.\n\n\n\n\n\n","category":"method"},{"location":"api/#Statistics.cov-Tuple{GaussianSystem}","page":"Library Reference","title":"Statistics.cov","text":"cov(Σ::GaussianSystem)\n\nGet the covariance matrix of Σ.\n\n\n\n\n\n","category":"method"},{"location":"api/#AlgebraicInference.invcov-Tuple{GaussianSystem}","page":"Library Reference","title":"AlgebraicInference.invcov","text":"invcov(Σ::GaussianSystem)\n\nGet the precision matrix of Σ.\n\n\n\n\n\n","category":"method"},{"location":"api/#Statistics.var-Tuple{GaussianSystem}","page":"Library Reference","title":"Statistics.var","text":"var(Σ::GaussianSystem)\n\nGet the variances of Σ.\n\n\n\n\n\n","category":"method"},{"location":"api/#Statistics.mean-Tuple{GaussianSystem}","page":"Library Reference","title":"Statistics.mean","text":"mean(Σ::GaussianSystem)\n\nGet the mean vector of Σ.\n\n\n\n\n\n","category":"method"},{"location":"api/#Catlab.WiringDiagrams.WiringDiagramAlgebras.oapply-Tuple{AbstractUWD, AbstractVector{<:GaussianSystem}}","page":"Library Reference","title":"Catlab.WiringDiagrams.WiringDiagramAlgebras.oapply","text":"oapply(wd::AbstractUWD, systems::AbstractVector{<:GaussianSystem})\n\nCompose Gaussian systems according to the undirected wiring diagram wd.\n\n\n\n\n\n","category":"method"},{"location":"api/#Problems","page":"Library Reference","title":"Problems","text":"","category":"section"},{"location":"api/","page":"Library Reference","title":"Library Reference","text":"InferenceProblem\nMinDegree\nMinFill\n\nInferenceProblem{T}(::AbstractUWD, ::AbstractDict, ::Union{Nothing, AbstractDict}) where T\nInferenceProblem{T}(::AbstractUWD, ::AbstractVector, ::Union{Nothing, AbstractVector}) where T\n\nsolve(::InferenceProblem, alg)\ninit(::InferenceProblem, alg)","category":"page"},{"location":"api/#AlgebraicInference.InferenceProblem","page":"Library Reference","title":"AlgebraicInference.InferenceProblem","text":"InferenceProblem{T₁, T₂}\n\nAn inference problem over a valuation algebra. Construct a solver for an inference problem with the function init, or solve it directly with solve.\n\n\n\n\n\n","category":"type"},{"location":"api/#AlgebraicInference.MinDegree","page":"Library Reference","title":"AlgebraicInference.MinDegree","text":"MinDegree\n\nContructs a covering join tree for an inference problem using the variable elimination algorithm. Variables are eliminated according to the \"minimum degree\" heuristic.\n\n\n\n\n\n","category":"type"},{"location":"api/#AlgebraicInference.MinFill","page":"Library Reference","title":"AlgebraicInference.MinFill","text":"MinFill\n\nContructs a covering join tree for an inference problem using the variable elimination algorithm. Variables are eliminated according to the \"minimum fill\" heuristic.\n\n\n\n\n\n","category":"type"},{"location":"api/#AlgebraicInference.InferenceProblem-Union{Tuple{T}, Tuple{AbstractUWD, AbstractDict, Union{Nothing, AbstractDict}}} where T","page":"Library Reference","title":"AlgebraicInference.InferenceProblem","text":"InferenceProblem{T}(wd::AbstractUWD, hom_map::AbstractDict,\n    ob_map::Union{Nothing, AbstractDict}=nothing;\n    hom_attr=:name, ob_attr=:variable) where T\n\nConstruct an inference problem that performs undirected composition. Before being composed, the values of hom_map are converted to type T.\n\n\n\n\n\n","category":"method"},{"location":"api/#AlgebraicInference.InferenceProblem-Union{Tuple{T}, Tuple{AbstractUWD, AbstractVector, Union{Nothing, AbstractVector}}} where T","page":"Library Reference","title":"AlgebraicInference.InferenceProblem","text":"InferenceProblem{T}(wd::AbstractUWD, homs::AbstractVector,\n    obs::Union{Nothing, AbstractVector}=nothing) where T\n\nConstruct an inference problem that performs undirected composition. Before being composed, the elements of homs are converted to type T.\n\n\n\n\n\n","category":"method"},{"location":"api/#CommonSolve.solve-Tuple{InferenceProblem, Any}","page":"Library Reference","title":"CommonSolve.solve","text":"solve(ip::InferenceProblem, alg)\n\nSolve an inference problem. The options for alg are\n\nMinDegree()\nMinFill()\n\n\n\n\n\n","category":"method"},{"location":"api/#CommonSolve.init-Tuple{InferenceProblem, Any}","page":"Library Reference","title":"CommonSolve.init","text":"init(ip::InferenceProblem, alg)\n\nConstruct a solver for an inference problem. The options for alg are\n\nMinDegree()\nMinFill()\n\n\n\n\n\n","category":"method"},{"location":"api/#Solvers","page":"Library Reference","title":"Solvers","text":"","category":"section"},{"location":"api/","page":"Library Reference","title":"Library Reference","text":"InferenceSolver\n\nsolve(::InferenceSolver)\nsolve!(::InferenceSolver)","category":"page"},{"location":"api/#AlgebraicInference.InferenceSolver","page":"Library Reference","title":"AlgebraicInference.InferenceSolver","text":"InferenceSolver{T₁, T₂}\n\nThis is the type constructed by init(ip::InferenceProblem). Use it with solve or solve! to solve inference problems.\n\nAn InferenceSolver can be reused to answer multiple queries:\n\nis = init(ip)\nsol1 = solve(is)\nis.query = query2\nsol2 = solve(is)\n\n\n\n\n\n","category":"type"},{"location":"api/#CommonSolve.solve-Tuple{InferenceSolver}","page":"Library Reference","title":"CommonSolve.solve","text":"solve(is::InferenceSolver)\n\nSolve an inference problem.\n\n\n\n\n\n","category":"method"},{"location":"api/#CommonSolve.solve!-Tuple{InferenceSolver}","page":"Library Reference","title":"CommonSolve.solve!","text":"solve!(is::InferenceSolver)\n\nSolve an inference problem, caching intermediate computations.\n\n\n\n\n\n","category":"method"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"EditURL = \"https://github.com/samuelsonric/AlgebraicInference.jl/blob/master/docs/literate/regression.jl\"","category":"page"},{"location":"generated/regression/#Linear-Regression","page":"Linear Regression","title":"Linear Regression","text":"","category":"section"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"using AlgebraicInference\nusing Catlab.Graphics, Catlab.Programs\nusing FillArrays\nusing LinearAlgebra\nusing StatsPlots","category":"page"},{"location":"generated/regression/#Frequentist-Linear-Regression","page":"Linear Regression","title":"Frequentist Linear Regression","text":"","category":"section"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"Consider the Gauss-Markov linear model","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"    y = X beta + epsilon","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"where X is an n times m matrix, beta is an m times 1 vector, and epsilon is an n times 1 normally distributed random vector with mean mathbf0 and covariance W. If X has full column rank, then the best linear unbiased estimator for beta is the random vector","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"    hatbeta = X^+ (I - (Q W Q)^+ Q W)^mathsfT y","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"where X^+ is the Moore-Penrose psuedoinverse of X, and","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"Q = I - X X^+","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"References:","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"Albert, Arthur. \"The Gauss-Markov Theorem for Regression Models with Possibly Singular Covariances.\" SIAM Journal on Applied Mathematics, vol. 24, no. 2, 1973, pp. 182–87.","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"X = [\n    1 0\n    0 1\n    0 0\n]\n\nW = [\n    1 0 0\n    0 1 0\n    0 0 1\n]\n\ny = [\n    1\n    1\n    1\n]\n\nQ = I - X * pinv(X)\nβ̂ = pinv(X) * (I - pinv(Q * W * Q) * Q * W)' * y","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"To solve for hatbeta using AlgebraicInference.jl, we construct an undirected wiring diagram.","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"wd = @relation (a₁, a₂) begin\n    X(a₁, a₂, b₁, b₂, b₃)\n    +(b₁, b₂, b₃, c₁, c₂, c₃, d₁, d₂, d₃)\n    ϵ(c₁, c₂, c₃)\n    y(d₁, d₂, d₃)\nend\n\nto_graphviz(wd; box_labels=:name, implicit_junctions=true)","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"Then we assign values to the boxes in wd and compute the result.","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"P = [\n    1 0 0 1 0 0\n    0 1 0 0 1 0\n    0 0 1 0 0 1\n]\n\nhm = Dict(\n    :X => kernel(X, Zeros(3), Zeros(3, 3)),\n    :+ => kernel(P, Zeros(3), Zeros(3, 3)),\n    :ϵ => normal(Zeros(3), W),\n    :y => normal(y, Zeros(3, 3)))\n\nβ̂ = mean(oapply(wd, hm))","category":"page"},{"location":"generated/regression/#Bayesian-Linear-Regression","page":"Linear Regression","title":"Bayesian Linear Regression","text":"","category":"section"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"Let rho = mathcalN(m V) be our prior belief about beta. Then our posterior belief hatrho is a bivariate normal distribution with mean","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"  hatm = m - V X^mathsfT (X V X^mathsfT + W)^+ (X m - y)","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"and covariance","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"  hatV = V - V X^mathsfT (X V X^mathsfT + W)^+ X V","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"V = [\n    1 0\n    0 1\n]\n\nm = [\n    0\n    0\n]\n\nm̂ = m - V * X' * pinv(X * V * X' + W) * (X * m - y)","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"V̂ = V - V * X' * pinv(X * V * X' + W) * X * V","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"To solve for hatrho using AlgebraicInference.jl, we construct an undirected wiring diagram.","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"wd = @relation (a₁, a₂) begin\n    ρ(a₁, a₂)\n    X(a₁, a₂, b₁, b₂, b₃)\n    +(b₁, b₂, b₃, c₁, c₂, c₃, d₁, d₂, d₃)\n    ϵ(c₁, c₂, c₃)\n    y(d₁, d₂, d₃)\nend\n\nto_graphviz(wd; box_labels=:name, implicit_junctions=true)","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"Then we assign values to the boxes in wd and compute the result.","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"hm = Dict(\n    :ρ => normal(m, V),\n    :X => kernel(X, Zeros(3), Zeros(3, 3)),\n    :+ => kernel(P, Zeros(3), Zeros(3, 3)),\n    :ϵ => normal(Zeros(3), W),\n    :y => normal(y, Zeros(3, 3)))\n\nm̂ = mean(oapply(wd, hm))","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"V̂ = cov(oapply(wd, hm))","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"covellipse!(m, V, aspect_ratio=:equal, label=\"prior\")\ncovellipse!(m̂, V̂, aspect_ratio=:equal, label=\"posterior\")","category":"page"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"EditURL = \"https://github.com/samuelsonric/AlgebraicInference.jl/blob/master/docs/literate/kalman.jl\"","category":"page"},{"location":"generated/kalman/#Kalman-Filter","page":"Kalman Filter","title":"Kalman Filter","text":"","category":"section"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"using AlgebraicInference\nusing BenchmarkTools\nusing Catlab.Graphics, Catlab.Programs, Catlab.WiringDiagrams\nusing Catlab.WiringDiagrams.MonoidalUndirectedWiringDiagrams: UntypedHypergraphDiagram\nusing Distributions\nusing FillArrays\nusing LinearAlgebra\nusing Random","category":"page"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"A Kalman filter with n steps is a probability distribution over states (s_1 dots s_n) and measurements (z_1 dots z_n) determined by the equations","category":"page"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"    s_i+1 mid s_i sim mathcalN(As_i P)","category":"page"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"and","category":"page"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"    z_i mid s_i sim mathcalN(Bs_i Q)","category":"page"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"θ = π / 15\n\nA = [\n    cos(θ) -sin(θ)\n    sin(θ) cos(θ)\n]\n\nB = [\n    1.3 0.0\n    0.0 0.7\n]\nP = [\n    0.05 0.0\n    0.0 0.05\n]\n\nQ = [\n    10.0 0.0\n    0.0 10.0\n]\n\nfunction generate_data(n; seed=42)\n    Random.seed!(seed)\n    x = zeros(2)\n    data = Vector{Float64}[]\n\n    for i in 1:n\n        x = rand(MvNormal(A * x, P))\n        push!(data, rand(MvNormal(B * x, Q)))\n    end\n\n    data\nend;\nnothing #hide","category":"page"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"The filtering problem involves predicting the value of the state s_n given observations of (z_1 dots z_n). The function kalman constructs a wiring diagram that represents the filtering problem.","category":"page"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"function kalman_step(i)\n    kf = UntypedHypergraphDiagram{String}(2)\n    add_box!(kf, 2; name=\"state\")\n    add_box!(kf, 4; name=\"predict\")\n    add_box!(kf, 4; name=\"measure\")\n    add_box!(kf, 2; name=\"z$i\")\n\n    add_wires!(kf, [\n        (0, 1) => (2, 3),\n        (0, 2) => (2, 4),\n        (1, 1) => (2, 1),\n        (1, 1) => (3, 1),\n        (1, 2) => (2, 2),\n        (1, 2) => (3, 2),\n        (3, 3) => (4, 1),\n        (3, 4) => (4, 2)])\n\n    kf\nend\n\nfunction kalman(n)\n    reduce((kf, i) -> ocompose(kalman_step(i), 1, kf), 2:n; init=kalman_step(1))\nend\n\nto_graphviz(kalman(5), box_labels=:name; implicit_junctions=true)","category":"page"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"We generate 100 points of data and solve the filtering problem.","category":"page"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"n = 100; kf = kalman(n); data = generate_data(n)\n\ndm = Dict(\"z$i\" => normal(data[i], Zeros(2, 2)) for i in 1:n)\n\nhm = Dict(\n    dm...,\n    \"state\" => normal(Zeros(2), 100I(2)),\n    \"predict\" => kernel(A, Zeros(2), P),\n    \"measure\" => kernel(B, Zeros(2), Q))\n\nmean(oapply(kf, hm))","category":"page"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"@benchmark oapply(kf, hm)","category":"page"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"Since the filtering problem is large, we may wish to solve it using belief propagation.","category":"page"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"ip = InferenceProblem{DenseGaussianSystem{Float64}}(kf, hm)\nis = init(ip, MinFill())\n\nmean(solve(is))","category":"page"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"@benchmark solve(is)","category":"page"},{"location":"#AlgebraicInference.jl","page":"AlgebraicInference.jl","title":"AlgebraicInference.jl","text":"","category":"section"},{"location":"","page":"AlgebraicInference.jl","title":"AlgebraicInference.jl","text":"AlgebraicInference.jl is a library for performing Bayesian inference on wiring diagrams,  building on Catlab.jl.","category":"page"},{"location":"#Gaussian-Systems","page":"AlgebraicInference.jl","title":"Gaussian Systems","text":"","category":"section"},{"location":"","page":"AlgebraicInference.jl","title":"AlgebraicInference.jl","text":"Gaussian systems were introduced by Jan Willems in his 2013 article Open Stochastic Systems. A probability space Sigma = (mathbbR^n mathcalE P) is called an n-variate Gaussian system with fiber mathbbL subseteq mathbbR^n if it is isomorphic to a Gaussian measure on the quotient space mathbbR^n  mathbbL.","category":"page"},{"location":"","page":"AlgebraicInference.jl","title":"AlgebraicInference.jl","text":"If mathbbL = 0, then Sigma is an n-variate normal distribution.","category":"page"},{"location":"","page":"AlgebraicInference.jl","title":"AlgebraicInference.jl","text":"Every n-variate Gaussian system Sigma corresponds to a convex energy function  E mathbbR^n to (-infty infty of the form","category":"page"},{"location":"","page":"AlgebraicInference.jl","title":"AlgebraicInference.jl","text":"    E(x) = begincases\n        frac12 x^mathsfT P x - x^mathsfT p  Sx = s \n        infty                                         textelse\n    endcases","category":"page"},{"location":"","page":"AlgebraicInference.jl","title":"AlgebraicInference.jl","text":"where P and S are positive semidefinite matrices, p in mathttimage(P), and s in mathttimage(S).","category":"page"},{"location":"","page":"AlgebraicInference.jl","title":"AlgebraicInference.jl","text":"If Sigma is an n-variate normal distribution, then E is its negative log-density.","category":"page"},{"location":"#Hypergraph-Categories","page":"AlgebraicInference.jl","title":"Hypergraph Categories","text":"","category":"section"},{"location":"","page":"AlgebraicInference.jl","title":"AlgebraicInference.jl","text":"There exists a hypergraph PROP whose morphisms m to n are m + n-variate Gaussian systems. Hence, Gaussian systems can be composed using undirected wiring diagrams.","category":"page"},{"location":"","page":"AlgebraicInference.jl","title":"AlgebraicInference.jl","text":"(Image: inference)","category":"page"},{"location":"","page":"AlgebraicInference.jl","title":"AlgebraicInference.jl","text":"These wiring diagrams look a lot like undirected graphical models. One difference is that wiring diagrams can contain half-edges, which specify which variables are marginalized out during composition. Hence, a wiring diagram can be thought of as an inference problem: a graphical model paired with a query.","category":"page"},{"location":"#Message-Passing","page":"AlgebraicInference.jl","title":"Message Passing","text":"","category":"section"},{"location":"","page":"AlgebraicInference.jl","title":"AlgebraicInference.jl","text":"Bayesian inference problems on large graphs are often solved using message passing. With AlgebraicInference.jl you can compose large numbers of Gaussian systems by translating undirected wiring diagrams into inference problems over a valuation algebra. These problems can be solved using generic inference algorithms like the Shenoy-Shafer architecture.","category":"page"}]
}

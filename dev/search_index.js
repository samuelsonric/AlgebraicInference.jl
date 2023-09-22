var documenterSearchIndex = {"docs":
[{"location":"api/","page":"Library Reference","title":"Library Reference","text":"e# Library Reference","category":"page"},{"location":"api/#Systems","page":"Library Reference","title":"Systems","text":"","category":"section"},{"location":"api/","page":"Library Reference","title":"Library Reference","text":"GaussianSystem\nCanonicalForm\nDenseGaussianSystem\nDenseCanonicalForm\n\nGaussianSystem(::AbstractMatrix, ::AbstractMatrix, ::AbstractVector, ::AbstractVector, ::Real)\nCanonicalForm(::AbstractMatrix, ::AbstractVector)\n\nnormal\nkernel\n\nlength(::GaussianSystem)\ncov(::GaussianSystem)\ninvcov(::GaussianSystem)\nvar(::GaussianSystem)\nmean(::GaussianSystem)","category":"page"},{"location":"api/#AlgebraicInference.GaussianSystem","page":"Library Reference","title":"AlgebraicInference.GaussianSystem","text":"GaussianSystem{T₁, T₂, T₃, T₄, T₅}\n\nA Gaussian system.\n\n\n\n\n\n","category":"type"},{"location":"api/#AlgebraicInference.CanonicalForm","page":"Library Reference","title":"AlgebraicInference.CanonicalForm","text":"CanonicalForm{T₁, T₂} = GaussianSystem{\n    T₁,\n    ZerosMatrix{Bool, Tuple{OneTo{Int}, OneTo{Int}}},\n    T₂,\n    ZerosVector{Bool, Tuple{OneTo{Int}}},\n    Bool}\n\nA canonical form.\n\n\n\n\n\n","category":"type"},{"location":"api/#AlgebraicInference.DenseGaussianSystem","page":"Library Reference","title":"AlgebraicInference.DenseGaussianSystem","text":"DenseGaussianSystem{T} = GaussianSystem{\n    Matrix{T},\n    Matrix{T},\n    Vector{T},\n    Vector{T},\n    T}\n\nA Gaussian system represented by dense matrices and vectors.\n\n\n\n\n\n","category":"type"},{"location":"api/#AlgebraicInference.DenseCanonicalForm","page":"Library Reference","title":"AlgebraicInference.DenseCanonicalForm","text":"DenseCanonicalForm{T} = CanonicalForm{\n    Matrix{T},\n    Vector{T}}\n\nA canonical form represented by a dense matrix and vector.\n\n\n\n\n\n","category":"type"},{"location":"api/#AlgebraicInference.GaussianSystem-Tuple{AbstractMatrix, AbstractMatrix, AbstractVector, AbstractVector, Real}","page":"Library Reference","title":"AlgebraicInference.GaussianSystem","text":"GaussianSystem(\n    P::AbstractMatrix,\n    S::AbstractMatrix,\n    p::AbstractVector,\n    s::AbstractVector,\n    σ::Real)\n\nConstruct a Gaussian system by specifying its energy function. \n\nYou should set σ equal to s^mathsfT S^+ s, where S^+ is the Moore-Penrose psuedoinverse of S.\n\n\n\n\n\n","category":"method"},{"location":"api/#AlgebraicInference.CanonicalForm-Tuple{AbstractMatrix, AbstractVector}","page":"Library Reference","title":"AlgebraicInference.CanonicalForm","text":"CanonicalForm(K::AbstractMatrix, h::AbstractVector)\n\nConstruct the canonical form mathcalC(K h g), where the normalization constant g is inferred from K and h.\n\n\n\n\n\n","category":"method"},{"location":"api/#AlgebraicInference.normal","page":"Library Reference","title":"AlgebraicInference.normal","text":"normal(μ::AbstractVector, Σ::AbstractMatrix)\n\nConstruct a multivariate normal distribution with mean vector μ and covariance matrix Σ.\n\n\n\n\n\nnormal(μ::Real, σ::Real)\n\nConstruct a normal distribution with mean μ and standard deviation σ.\n\n\n\n\n\n","category":"function"},{"location":"api/#AlgebraicInference.kernel","page":"Library Reference","title":"AlgebraicInference.kernel","text":"kernel(L::AbstractMatrix, μ::AbstractVector, Σ::AbstractMatrix)\n\nConstruct a conditional distribution of the form p(Y mid X = x) = mathcalN(Lx + mu Sigma)\n\n\n\n\n\nkernel(l::AbstractVector, μ::Real, σ::Real)\n\nConstruct a conditional distribution of the form p(Y mid X = x) = mathcalN(l^mathsfTx + mu sigma^2)\n\n\n\n\n\n","category":"function"},{"location":"api/#Base.length-Tuple{GaussianSystem}","page":"Library Reference","title":"Base.length","text":"length(Σ::GaussianSystem)\n\nGet the dimension of Σ.\n\n\n\n\n\n","category":"method"},{"location":"api/#Statistics.cov-Tuple{GaussianSystem}","page":"Library Reference","title":"Statistics.cov","text":"cov(Σ::GaussianSystem; atol::Real=1e-8)\n\nGet the covariance matrix of Σ.\n\n\n\n\n\n","category":"method"},{"location":"api/#Distributions.invcov-Tuple{GaussianSystem}","page":"Library Reference","title":"Distributions.invcov","text":"invcov(Σ::GaussianSystem)\n\nGet the precision matrix of Σ.\n\n\n\n\n\n","category":"method"},{"location":"api/#Statistics.var-Tuple{GaussianSystem}","page":"Library Reference","title":"Statistics.var","text":"var(Σ::GaussianSystem; atol::Real=1e-8)\n\nGet the variances of Σ.\n\n\n\n\n\n","category":"method"},{"location":"api/#Statistics.mean-Tuple{GaussianSystem}","page":"Library Reference","title":"Statistics.mean","text":"mean(Σ::GaussianSystem; atol::Real=1e-8)\n\nGet the mean vector of Σ.\n\n\n\n\n\n","category":"method"},{"location":"api/#Problems","page":"Library Reference","title":"Problems","text":"","category":"section"},{"location":"api/","page":"Library Reference","title":"Library Reference","text":"InferenceProblem\n\nInferenceProblem(::RelationDiagram, ::AbstractDict, ::AbstractDict)\nInferenceProblem(::BayesNet, ::AbstractVector, ::AbstractDict)\n\nsolve(::InferenceProblem, ::EliminationAlgorithm, ::SupernodeType, ::ArchitectureType)\ninit(::InferenceProblem, ::EliminationAlgorithm, ::SupernodeType, ::ArchitectureType)","category":"page"},{"location":"api/#AlgebraicInference.InferenceProblem","page":"Library Reference","title":"AlgebraicInference.InferenceProblem","text":"InferenceProblem{T₁, T₂, T₃, T₄}\n\nAn inference problem computes the conditional distribution of X given Y = e, where X and Y are random variables whose joint probability distribution is specified by a graphical model.\n\n\n\n\n\n","category":"type"},{"location":"api/#AlgebraicInference.InferenceProblem-Tuple{RelationDiagram, AbstractDict, AbstractDict}","page":"Library Reference","title":"AlgebraicInference.InferenceProblem","text":"InferenceProblem(\n    diagram::RelationDiagram,\n    hom_map::AbstractDict,\n    ob_map::AbstractDict;\n    hom_attr::Symbol=:name,\n    ob_attr::Symbol=:junction_type,\n    var_attr::Symbol=:variable\n    check::Bool=true)\n\nConstruct an inference problem that performs undirected compositon.\n\n\n\n\n\n","category":"method"},{"location":"api/#AlgebraicInference.InferenceProblem-Tuple{BayesNet, AbstractVector, AbstractDict}","page":"Library Reference","title":"AlgebraicInference.InferenceProblem","text":"InferenceProblem(\n    network::BayesNet,\n    query::AbstractVector,\n    context::AbstractDict)\n\nConstruct an inference problem that queries a Bayesian network.\n\n\n\n\n\n","category":"method"},{"location":"api/#CommonSolve.solve-Tuple{InferenceProblem, EliminationAlgorithm, SupernodeType, ArchitectureType}","page":"Library Reference","title":"CommonSolve.solve","text":"solve(\n    problem::InferenceProblem,\n    elimination_algorithm::EliminationAlgorithm=MinFill()\n    supernode_type::SupernodeType=Node()\n    architecture_type::ArchitectureType=ShenoyShafer())\n\nSolve an inference problem.\n\n\n\n\n\n","category":"method"},{"location":"api/#CommonSolve.init-Tuple{InferenceProblem, EliminationAlgorithm, SupernodeType, ArchitectureType}","page":"Library Reference","title":"CommonSolve.init","text":"init(\n    problem::InferenceProblem,\n    elimination_algorithm::EliminationAlgorithm=MinFill(),\n    supernode_type::SupernodeType=Node(),\n    architecture_type::ArchitectureType=ShenoyShafer())\n\nConstruct a solver for an inference problem.\n\n\n\n\n\n","category":"method"},{"location":"api/#Solvers","page":"Library Reference","title":"Solvers","text":"","category":"section"},{"location":"api/","page":"Library Reference","title":"Library Reference","text":"InferenceSolver\n\nsolve!(::InferenceSolver)\nmean(::InferenceSolver)\nrand(::AbstractRNG, ::InferenceSolver)","category":"page"},{"location":"api/#AlgebraicInference.InferenceSolver","page":"Library Reference","title":"AlgebraicInference.InferenceSolver","text":"InferenceSolver{T₁, T₂, T₃, T₄, T₅}\n\nA solver for an inference problem. \n\nAn InferenceSolver can be reused to answer multiple queries:\n\nis = init(ip)\nsol₁ = solve(is)\nis.query = query\nsol₂ = solve(is)\n\n\n\n\n\n","category":"type"},{"location":"api/#CommonSolve.solve!-Tuple{InferenceSolver}","page":"Library Reference","title":"CommonSolve.solve!","text":"solve!(solver::InferenceSolver)\n\nSolve an inference problem, caching intermediate computations.\n\n\n\n\n\n","category":"method"},{"location":"api/#Statistics.mean-Tuple{InferenceSolver}","page":"Library Reference","title":"Statistics.mean","text":"mean(solver::InferenceSolver)\n\n\n\n\n\n","category":"method"},{"location":"api/#Base.rand-Tuple{AbstractRNG, InferenceSolver}","page":"Library Reference","title":"Base.rand","text":"rand(rng::AbstractRNG=default_rng(), solver::InferenceSolver)\n\n\n\n\n\n","category":"method"},{"location":"api/#Elimination","page":"Library Reference","title":"Elimination","text":"","category":"section"},{"location":"api/","page":"Library Reference","title":"Library Reference","text":"EliminationAlgorithm\nMaxCardinality\nMinDegree\nMinFill\nChordalGraph\nCuthillMcKeeJL_RCM\nAMDJL_AMD\nMetisJL_ND\n\nSupernodeType\nNode\nMaximalSupernode","category":"page"},{"location":"api/#AlgebraicInference.EliminationAlgorithm","page":"Library Reference","title":"AlgebraicInference.EliminationAlgorithm","text":"EliminationAlgorithm\n\nAn algorithm that computes an elimination order for an undirected graph.\n\n\n\n\n\n","category":"type"},{"location":"api/#AlgebraicInference.MaxCardinality","page":"Library Reference","title":"AlgebraicInference.MaxCardinality","text":"MaxCardinality <: EliminationAlgorithm\n\nThe maximum cardinality search algorithm.\n\n\n\n\n\n","category":"type"},{"location":"api/#AlgebraicInference.MinDegree","page":"Library Reference","title":"AlgebraicInference.MinDegree","text":"MinDegree <: EliminationAlgorithm\n\nThe minimum-degree heuristic.\n\n\n\n\n\n","category":"type"},{"location":"api/#AlgebraicInference.MinFill","page":"Library Reference","title":"AlgebraicInference.MinFill","text":"MinFill <: EliminationAlgorithm\n\nThe minimum-fill heuristic.\n\n\n\n\n\n","category":"type"},{"location":"api/#AlgebraicInference.ChordalGraph","page":"Library Reference","title":"AlgebraicInference.ChordalGraph","text":"ChordalGraph <: EliminationAlgorithm\n\nAn efficient algorithm for chordal graphs.\n\n\n\n\n\n","category":"type"},{"location":"api/#AlgebraicInference.CuthillMcKeeJL_RCM","page":"Library Reference","title":"AlgebraicInference.CuthillMcKeeJL_RCM","text":"CuthillMcKeeJL_RCM <: EliminationAlgorithm\n\nThe reverse Cuthill-McKee algorithm. Uses CuthillMckee.jl.\n\n\n\n\n\n","category":"type"},{"location":"api/#AlgebraicInference.AMDJL_AMD","page":"Library Reference","title":"AlgebraicInference.AMDJL_AMD","text":"AMDJL_AMD <: EliminationAlgorithm\n\nThe approximate minimum degree algorithm. Uses AMD.jl.\n\n\n\n\n\n","category":"type"},{"location":"api/#AlgebraicInference.MetisJL_ND","page":"Library Reference","title":"AlgebraicInference.MetisJL_ND","text":"MetisJL_ND <: EliminationAlgorithm\n\nThe nested dissection heuristic. Uses Metis.jl.\n\n\n\n\n\n","category":"type"},{"location":"api/#AlgebraicInference.SupernodeType","page":"Library Reference","title":"AlgebraicInference.SupernodeType","text":"SupernodeType\n\nA type of supernode.\n\n\n\n\n\n","category":"type"},{"location":"api/#AlgebraicInference.Node","page":"Library Reference","title":"AlgebraicInference.Node","text":"Node <: SupernodeType\n\nThe single-vertex supernode.\n\n\n\n\n\n","category":"type"},{"location":"api/#AlgebraicInference.MaximalSupernode","page":"Library Reference","title":"AlgebraicInference.MaximalSupernode","text":"MaximalSupernode <: SupernodeType\n\nThe maximal supernode.\n\n\n\n\n\n","category":"type"},{"location":"api/#Architectures","page":"Library Reference","title":"Architectures","text":"","category":"section"},{"location":"api/","page":"Library Reference","title":"Library Reference","text":"ArchitectureType\nShenoyShafer\nLauritzenSpiegelhalter","category":"page"},{"location":"api/#AlgebraicInference.ArchitectureType","page":"Library Reference","title":"AlgebraicInference.ArchitectureType","text":"ArchitectureType\n\nAn algorithm that computes marginal distributions by passing messages over a join tree.\n\n\n\n\n\n","category":"type"},{"location":"api/#AlgebraicInference.ShenoyShafer","page":"Library Reference","title":"AlgebraicInference.ShenoyShafer","text":"ShenoyShafer <: ArchitectureType\n\nThe Shenoy-Shafer architecture.\n\n\n\n\n\n","category":"type"},{"location":"api/#AlgebraicInference.LauritzenSpiegelhalter","page":"Library Reference","title":"AlgebraicInference.LauritzenSpiegelhalter","text":"LauritzenSpiegelhalter <: ArchitectureType\n\nThe Lauritzen-Spiegelhalter architecture.\n\n\n\n\n\n","category":"type"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"EditURL = \"../../literate/regression.jl\"","category":"page"},{"location":"generated/regression/#Linear-Regression","page":"Linear Regression","title":"Linear Regression","text":"","category":"section"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"using AlgebraicInference\nusing Catlab.Graphics, Catlab.Programs\nusing FillArrays\nusing LinearAlgebra\nusing StatsPlots","category":"page"},{"location":"generated/regression/#Frequentist-Linear-Regression","page":"Linear Regression","title":"Frequentist Linear Regression","text":"","category":"section"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"Consider the Gauss-Markov linear model","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"    y = X beta + epsilon","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"where X is an n times m matrix, beta is an m times 1 vector, and epsilon is an n times 1 normally distributed random vector with mean mathbf0 and covariance W. If X has full column rank, then the best linear unbiased estimator for beta is the random vector","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"    hatbeta = X^+ (I - (Q W Q)^+ Q W)^mathsfT y","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"where X^+ is the Moore-Penrose psuedoinverse of X, and","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"Q = I - X X^+","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"References:","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"Albert, Arthur. \"The Gauss-Markov Theorem for Regression Models with Possibly Singular Covariances.\" SIAM Journal on Applied Mathematics, vol. 24, no. 2, 1973, pp. 182–87.","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"X = [\n    1  0\n    0  1\n    0  0\n]\n\nW = [\n    0  0  0\n    0  1 .5\n    0 .5  1\n]\n\ny = [\n    1\n    1\n    1\n]\n\nQ = I - X * pinv(X)\nβ̂ = pinv(X) * (I - pinv(Q * W * Q) * Q * W)' * y\nround.(β̂; digits=4)","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"To solve for hatbeta using AlgebraicInference.jl, we construct an undirected wiring diagram.","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"wd = @relation (a,) where (a::m, b::n, c::n, d::n) begin\n    X(a, b)\n    +(b, c, d)\n    ϵ(c)\n    y(d)\nend\n\nto_graphviz(wd; box_labels=:name, implicit_junctions=true)","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"Then we assign values to the boxes in wd and compute the result.","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"P = [\n    1  0  0  1  0  0\n    0  1  0  0  1  0\n    0  0  1  0  0  1\n]\n\nhom_map = Dict{Symbol, DenseGaussianSystem{Float64}}(\n    :X => kernel(X, Zeros(3), Zeros(3, 3)),\n    :+ => kernel(P, Zeros(3), Zeros(3, 3)),\n    :ϵ => normal(Zeros(3), W),\n    :y => normal(y, Zeros(3, 3)))\n\nob_map = Dict(\n    :m => 2,\n    :n => 3)\n\nproblem = InferenceProblem(wd, hom_map, ob_map)\n\nΣ̂ = solve(problem)\n\nβ̂ = mean(Σ̂)\n\nround.(β̂; digits=4)","category":"page"},{"location":"generated/regression/#Bayesian-Linear-Regression","page":"Linear Regression","title":"Bayesian Linear Regression","text":"","category":"section"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"Let rho = mathcalN(m V) be our prior belief about beta. Then our posterior belief hatrho is a bivariate normal distribution with mean","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"  hatm = m - V X^mathsfT (X V X^mathsfT + W)^+ (X m - y)","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"and covariance","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"  hatV = V - V X^mathsfT (X V X^mathsfT + W)^+ X V","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"V = [\n    1  0\n    0  1\n]\n\nm = [\n    0\n    0\n]\n\nm̂ = m - V * X' * pinv(X * V * X' + W) * (X * m - y)\n\nround.(m̂; digits=4)","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"V̂ = V - V * X' * pinv(X * V * X' + W) * X * V\n\nround.(V̂; digits=4)","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"To solve for hatrho using AlgebraicInference.jl, we construct an undirected wiring diagram.","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"wd = @relation (a,) where (a::m, b::n, c::n, d::n) begin\n    ρ(a)\n    X(a, b)\n    +(b, c, d)\n    ϵ(c)\n    y(d)\nend\n\nto_graphviz(wd; box_labels=:name, implicit_junctions=true)","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"Then we assign values to the boxes in wd and compute the result.","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"hom_map = Dict{Symbol, DenseGaussianSystem{Float64}}(\n    :ρ => normal(m, V),\n    :X => kernel(X, Zeros(3), Zeros(3, 3)),\n    :+ => kernel(P, Zeros(3), Zeros(3, 3)),\n    :ϵ => normal(Zeros(3), W),\n    :y => normal(y, Zeros(3, 3)))\n\nob_map = Dict(\n    :m => 2,\n    :n => 3)\n\nproblem = InferenceProblem(wd, hom_map, ob_map)\n\nΣ̂ = solve(problem)\n\nm̂ = mean(Σ̂)\n\nround.(m̂; digits=4)","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"V̂ = cov(Σ̂)\n\nround.(V̂; digits=4)","category":"page"},{"location":"generated/regression/","page":"Linear Regression","title":"Linear Regression","text":"plot()\ncovellipse!(m, V, aspect_ratio=:equal, label=\"prior\")\ncovellipse!(m̂, V̂, aspect_ratio=:equal, label=\"posterior\")","category":"page"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"EditURL = \"../../literate/kalman.jl\"","category":"page"},{"location":"generated/kalman/#Kalman-Filter","page":"Kalman Filter","title":"Kalman Filter","text":"","category":"section"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"using AlgebraicInference\nusing BenchmarkTools\nusing Catlab.Graphics, Catlab.Programs, Catlab.WiringDiagrams\nusing Distributions\nusing FillArrays\nusing LinearAlgebra\nusing Random\nusing StatsPlots","category":"page"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"A Kalman filter with n steps is a probability distribution over states (x_1 dots x_n) and measurements (y_1 dots y_n) determined by the equations","category":"page"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"    p(x_i+1 mid x_i) = mathcalN(Ax_i P)","category":"page"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"and","category":"page"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"    p(y_i mid x_i) = mathcalN(Bx_i Q)","category":"page"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"θ = π / 15\n\nA = [\n    cos(θ) -sin(θ)\n    sin(θ)  cos(θ)\n]\n\nB = [\n    1.3 0.0\n    0.0 0.7\n]\n\nP = [\n    0.05 0.0\n    0.0 0.05\n]\n\nQ = [\n    10.0 0.0\n    0.0 10.0\n]\n\n\nfunction generate_data(n::Integer; seed::Integer=42)\n    Random.seed!(seed)\n    data = Matrix{Float64}(undef, 2, n)\n\n    N₁ = MvNormal(P)\n    N₂ = MvNormal(Q)\n\n    x = Zeros(2)\n\n    for i in 1:n\n        x = rand(N₁) + A * x\n        data[:, i] .= rand(N₂) + B * x\n    end\n\n    data\nend;\nnothing #hide","category":"page"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"The prediction problem involves finding the posterior mean and covariance of the state x_n + 1 given observations of (y_1 dots y_n).","category":"page"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"function make_diagram(n::Integer)\n    outer_ports = [\"X\"]\n\n    uwd = TypedRelationDiagram{String, String, Tuple{Int, Int}}(outer_ports)\n\n    x = add_junction!(uwd, \"X\"; variable=(1, 1))\n    y = add_junction!(uwd, \"Y\"; variable=(2, 1))\n\n    state   = add_box!(uwd, [\"X\"];      name=\"state\")\n    predict = add_box!(uwd, [\"X\", \"X\"]; name=\"predict\")\n    measure = add_box!(uwd, [\"X\", \"Y\"]; name=\"measure\")\n    context = add_box!(uwd, [\"Y\"];      name=\"y1\")\n\n    set_junction!(uwd, (state,   1), x)\n    set_junction!(uwd, (predict, 1), x)\n    set_junction!(uwd, (measure, 1), x)\n    set_junction!(uwd, (measure, 2), y)\n    set_junction!(uwd, (context, 1), y)\n\n    for i in 2:n\n        x = add_junction!(uwd, \"X\"; variable=(1, i))\n        y = add_junction!(uwd, \"Y\"; variable=(2, i))\n\n        set_junction!(uwd, (predict, 2), x)\n\n        predict = add_box!(uwd, [\"X\", \"X\"]; name=\"predict\")\n        measure = add_box!(uwd, [\"X\", \"Y\"]; name=\"measure\")\n        context = add_box!(uwd, [\"Y\"];      name=\"y$i\")\n\n        set_junction!(uwd, (predict, 1), x)\n        set_junction!(uwd, (measure, 1), x)\n        set_junction!(uwd, (measure, 2), y)\n        set_junction!(uwd, (context, 1), y)\n    end\n\n    i = n + 1\n    x = add_junction!(uwd, \"X\"; variable=(1, i))\n\n    set_junction!(uwd, (0,       1), x)\n    set_junction!(uwd, (predict, 2), x)\n\n    uwd\nend\n\nto_graphviz(make_diagram(5), box_labels=:name; junction_labels=:variable)","category":"page"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"We generate 100 points of data and solve the prediction problem.","category":"page"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"n = 100\n\ndiagram = make_diagram(n)\n\nhom_map = Dict{String, DenseGaussianSystem{Float64}}(\n    \"state\" => normal(Zeros(2), 100I(2)),\n    \"predict\" => kernel(A, Zeros(2), P),\n    \"measure\" => kernel(B, Zeros(2), Q))\n\nob_map = Dict(\n    \"X\" => 2,\n    \"Y\" => 2)\n\ndata = generate_data(n)\n\nfor i in 1:n\n    hom_map[\"y$i\"] = normal(data[:, i], Zeros(2, 2))\nend\n\nproblem = InferenceProblem(diagram, hom_map, ob_map)\n\nsolver = init(problem)\n\nΣ = solve!(solver)\n\nm = mean(Σ)\n\nround.(m; digits=4)","category":"page"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"V = cov(Σ)\n\nround.(V; digits=4)","category":"page"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"The smoothing problem involves finding the posterior means and covariances of the states (x_1 dots x_n - 1) given observations of (y_1 dots y_n).","category":"page"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"Calling mean(solver) computes a dictionary with the posterior mean of every variable in the model.","category":"page"},{"location":"generated/kalman/","page":"Kalman Filter","title":"Kalman Filter","text":"ms = mean(solver)\n\nx = Matrix{Float64}(undef, 2, n)\ny = Matrix{Float64}(undef, 2, n)\n\nfor i in 1:n\n    x[:, i] .= ms[1, i]\n    y[:, i] .= ms[2, i]\nend\n\nplot()\nplot!(x[1, :], label=\"x₁\")\nplot!(x[2, :], label=\"x₂\")","category":"page"},{"location":"#AlgebraicInference.jl","page":"AlgebraicInference.jl","title":"AlgebraicInference.jl","text":"","category":"section"},{"location":"","page":"AlgebraicInference.jl","title":"AlgebraicInference.jl","text":"AlgebraicInference.jl is a library for performing Bayesian inference on wiring diagrams,  building on Catlab.jl.","category":"page"},{"location":"#Gaussian-Systems","page":"AlgebraicInference.jl","title":"Gaussian Systems","text":"","category":"section"},{"location":"","page":"AlgebraicInference.jl","title":"AlgebraicInference.jl","text":"Gaussian systems were introduced by Jan Willems in his 2013 article Open Stochastic Systems. A probability space Sigma = (mathbbR^n mathcalE P) is called an n-variate Gaussian system with fiber mathbbL subseteq mathbbR^n if it is isomorphic to a Gaussian measure on the quotient space mathbbR^n  mathbbL.","category":"page"},{"location":"","page":"AlgebraicInference.jl","title":"AlgebraicInference.jl","text":"If mathbbL = 0, then Sigma is an n-variate normal distribution.","category":"page"},{"location":"","page":"AlgebraicInference.jl","title":"AlgebraicInference.jl","text":"Every n-variate Gaussian system Sigma corresponds to a convex energy function  E mathbbR^n to (-infty infty of the form","category":"page"},{"location":"","page":"AlgebraicInference.jl","title":"AlgebraicInference.jl","text":"    E(x) = begincases\n        frac12 x^mathsfT P x - x^mathsfT p  Sx = s \n        infty                                         textelse\n    endcases","category":"page"},{"location":"","page":"AlgebraicInference.jl","title":"AlgebraicInference.jl","text":"where P and S are positive semidefinite matrices, p in mathttimage(P), and s in mathttimage(S).","category":"page"},{"location":"","page":"AlgebraicInference.jl","title":"AlgebraicInference.jl","text":"If Sigma is an n-variate normal distribution, then E is its negative log-density.","category":"page"},{"location":"#Hypergraph-Categories","page":"AlgebraicInference.jl","title":"Hypergraph Categories","text":"","category":"section"},{"location":"","page":"AlgebraicInference.jl","title":"AlgebraicInference.jl","text":"There exists a hypergraph PROP whose morphisms m to n are m + n-variate Gaussian systems. Hence, Gaussian systems can be composed using undirected wiring diagrams.","category":"page"},{"location":"","page":"AlgebraicInference.jl","title":"AlgebraicInference.jl","text":"(Image: inference)","category":"page"},{"location":"","page":"AlgebraicInference.jl","title":"AlgebraicInference.jl","text":"These wiring diagrams look a lot like undirected graphical models. One difference is that wiring diagrams can contain half-edges, which specify which variables are marginalized out during composition. Hence, a wiring diagram can be thought of as an inference problem: a graphical model paired with a query.","category":"page"},{"location":"#Message-Passing","page":"AlgebraicInference.jl","title":"Message Passing","text":"","category":"section"},{"location":"","page":"AlgebraicInference.jl","title":"AlgebraicInference.jl","text":"Bayesian inference problems on large graphs are often solved using message passing. With AlgebraicInference.jl you can compose large numbers of Gaussian systems by translating undirected wiring diagrams into inference problems over a valuation algebra. These problems can be solved using generic inference algorithms like the Shenoy-Shafer architecture.","category":"page"}]
}

using Random, Distributions, Interpolations, Base.Iterators, FastGaussQuadrature, BenchmarkTools, Optim, LogExpFunctions, Plots, CSV, DataFrames, DataFramesMeta, ForwardDiff, LinearAlgebra, Roots, QuadGK, Statistics, ThreadsX, InverseFunctions, FileIO, Images, StatsAPI, StatsBase, RegressionTables

const Z̄ = 1.9599639845401

@inline diffcdf(N,b,a) = cdf(N,b) - cdf(N,a)

@inline sqrtNaN(x) = x<0 ? typeof(x)(NaN) : sqrt(x)

import Base.rand, Distributions.pdf, Distributions.logpdf, Distributions.cdf, Distributions.logcdf, Distributions.ccdf, Distributions.logccdf, Statistics.quantile

# type to represent distribution of maximum of m normal draws, Float64 only
struct MaxNormal <: Distribution{Univariate, Continuous}
	m::Float64  # number of draws over which to take max
	invm::Float64
	μ::Float64  # mean
	σ::Float64  # sd
	𝒩::Normal{Float64}
	MaxNormal(m=1., μ=0., σ=1.) = new(m,1/m,μ,σ,Normal(μ,σ))
end
cdf(s::MaxNormal, x::Float64) = cdf(s.𝒩,x) ^ s.m
logccdf(s::MaxNormal, x::Float64) = log1mexp(s.m * logcdf(s.𝒩,x))
ccdf(s::MaxNormal, x::Float64) = exp(logccdf(s,x))
pdf(s::MaxNormal, x::Float64) = pdf(s.𝒩,x) * s.m * cdf(s.𝒩,x)^(s.m-1)
logpdf(s::MaxNormal, x::Float64) = logpdf(s.𝒩,x) + log(s.m) + (s.m-1) * logcdf(s.𝒩,x)
quantile(s::MaxNormal, x::Float64) = quantile(s.𝒩, x^s.invm)
rand(rng::AbstractRNG, s::MaxNormal) = quantile(s.𝒩, rand(rng)^s.invm)

struct MinNormal <: Distribution{Univariate, Continuous}
	m::Float64  # number of draws over which to take max
	invm::Float64
	μ::Float64  # mean
	σ::Float64  # sd
	𝒩::Normal{Float64}
	MinNormal(m=1., μ=0., σ=1.) = new(m,1/m,μ,σ,Normal(μ,σ))
end
cdf(s::MinNormal, x::Float64) = exp(logcdf(s,x))
logcdf(s::MinNormal, x::Float64) = log1mexp(s.m * logccdf(s.𝒩,x))
ccdf(s::MinNormal, x::Float64) = ccdf(s.𝒩,x) ^ s.m
pdf(s::MinNormal, x::Float64) = pdf(s.𝒩,x) * s.m * ccdf(s.𝒩,x)^(s.m-1)
logpdf(s::MinNormal, x::Float64) = logpdf(s.𝒩,x) + log(s.m) + (s.m-1) * logccdf(s.𝒩,x)
quantile(s::MinNormal, x::Float64) = cquantile(s.𝒩, (1-x)^s.invm)
rand(rng::AbstractRNG, s::MinNormal) = cquantile(s.𝒩, rand(rng)^s.invm)

# to parameterize an n-vector of probabilities summing to 1 with an unbounded (n-1)-vector, apply logistic transform to latter, then map to squared spherical coordinates
# https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates, https://math.stackexchange.com/questions/2861449/parameterizations-of-the-unit-simplex-in-mathbbr3
function RⁿtoSimplex(q::AbstractVector{T}) where T
	_q = logistic.(q)
	p = Vector{T}(undef, length(q)+1)
	Πsin² = one(T)
	for i ∈ eachindex(_q)
		cos² = cospi(_q[i])^2
		p[i] = Πsin² * cos²
		Πsin² *= one(T) - cos²
	end
	p[end] = Πsin²
	p
end
function SimplextoRⁿ(p::AbstractVector{T}) where T
	q = Vector{T}(undef, length(p)-1)
	sum = p[end]
	for i ∈ reverse(eachindex(q))
		sum += p[i]
		q[i] = acos(√(p[i] / sum)) / π
	end
	q .= logit.(q)
end
InverseFunctions.inverse(::typeof(SimplextoRⁿ)) = RⁿtoSimplex

get1(x) = x[1]
get2(x) = x[1]
get3(x) = x[1]
get4(x) = x[1]
const consvec = (get1, get2, get3, get4)
InverseFunctions.inverse(::typeof(get1)) = x->fill(x,1)
InverseFunctions.inverse(::typeof(get2)) = x->fill(x,2)
InverseFunctions.inverse(::typeof(get3)) = x->fill(x,3)
InverseFunctions.inverse(::typeof(get4)) = x->fill(x,4)

bcast(f) = Broadcast.BroadcastFunction(f)  # short-hand for forming the broadcasting version of a function, which works with InverseFunctions

# unlogged likelihood for a single observation. For graphs.
function HnFl(z; p::Vector, μ=0., τ::Vector, pF₀, pH₀, kH=[0.,0.], m=1, truncate=true, rtol=.01)
	iszero(length(μ)) && (μ = 0.)
	μ isa Number && (μ = fill(μ,length(τ)))
	pL₀,pU₀ = pH₀
	kL ,kU  = kH
	pL = pL₀ * exp(-kL*(Z̄+z))
	pU = pU₀ * exp(-kU*(Z̄-z))
	pF = pF₀ * (1 - pL - pU)
	pD = 1 - pL - pU - pF
	σ² = 1 .+ τ.^2
	𝒩  = Normal()
	𝒩μ = @. Normal(μ, √σ²)
	𝒩ω = @. NormalCanon(z + μ/τ^2, 1 + 1/τ^2)
	if abs(z) ≥ Z̄
		result = 0.
		@inbounds for (pᵢ,𝒩μᵢ,𝒩ωᵢ) ∈ zip(p,𝒩μ,𝒩ω)
			if z < 0.
				result += pᵢ * pdf(𝒩μᵢ, z) * (1 + m * pL₀ * exp(kL*(kL/2-Z̄)) * quadgk(ω -> (a = pdf(𝒩ωᵢ,ω) * exp(-kL*ω) * diffcdf(𝒩, Z̄-ω+kL, -Z̄-ω+kL) * ccdf(𝒩,z-ω)^(m-1) / (1-ccdf(𝒩, -Z̄-ω)^m);
																																		                 isnan(a) || isinf(a) ? 0. : a), 
																                                               -20, 20; rtol)[1])
			else
				result += pᵢ * pdf(𝒩μᵢ, z) * (1 + m * pU₀ * exp(kU*(kU/2-Z̄)) * quadgk(ω -> (a = pdf(𝒩ωᵢ,ω) * exp( kU*ω) * diffcdf(𝒩, Z̄-ω-kU, -Z̄-ω-kU) * cdf(𝒩,z-ω)^(m-1) / (1 - cdf(𝒩,  Z̄-ω)^m);
																																		                 isnan(a) || isinf(a) ? 0. : a), 
																                                               -20, 20; rtol)[1])
			end
		end
	else
		result = pD * dot(p, pdf.(𝒩μ, z))
	end
	truncate && (result /= 1 - pF₀ * dot(p, @. (diffcdf(𝒩μ, Z̄, -Z̄) - pL₀ * exp(kL*(σ²*kL/2-μ-Z̄)) * diffcdf(𝒩μ,  σ²*kL+Z̄,  σ²*kL-Z̄)
	                                                                - pU₀ * exp(kU*(σ²*kU/2+μ-Z̄)) * diffcdf(𝒩μ, -σ²*kU+Z̄, -σ²*kU-Z̄)  )))
	result
end

# object to hold pre-computed stuff for log likelihood computation
struct HnFmodel
	D::Int  # number of mixture components
	z::Vector{Float64}  # all data
	zC::Vector{Float64}  # just the central, insignificant z's
	Z̄pzC::Vector{Float64}; Z̄mzC::Vector{Float64}  # Z̄ .+ zC, Z̄ .- zC
	N::Int; NC::Int; NL::Int; NU::Int  # number of z's, insigficant z's, lower significant z's, upper
	knots::LinRange  # interpolation knots in [Z̄,max]
	spline::Interpolations.InterpolationType  # type of interpolation
	zLint::Vector{Float64}; zUint::Vector{Float64}  # lower- & upper-tail significant z values mapped to cardinal knot numbering space since interpolate() is faster with cardinally spaced knots
	X::Vector{Float64}; W::Vector{Float64}  # quadrature nodes & weights
	lnW::Vector{Float64}
	xforms::Dict{Symbol, Function}
end
# constructor
function HnFmodel(z::Vector{Float64}; D::Int, interpres::Int=100, quadnodes::Int=25, kwargs...)
	zC = z[abs.(z) .< Z̄]

	s = Z̄ - 3/interpres; e = max(10,maximum(z))+.2
	knots = s : 1/interpres : e  # LinRange(s, e, ceil(Int, (e - s) * interpres) + 1)
	zLint = @. (-z[z ≤ -Z̄] - s) * interpres + 1  # map tail z values to knot numbering 1, 2, ... for Z̄-3/interpres, Z̄-2/interpres, ...
	zUint = @. ( z[z ≥  Z̄] - s) * interpres + 1

	X, W = gausshermite(quadnodes)
	W ./= √π

	HnFmodel(D, z, zC, Z̄.+zC, Z̄.-zC, length(z), length(zC), length(zLint), length(zUint), knots, BSpline(Linear()), zLint, zUint, X, W, log.(W), Dict(kwargs))
end

# bulk log probabilities as function of data & parameters, for estimation
function HnFll(M::HnFmodel; p::AbstractVector{<:Real}, μ=0., τ::AbstractVector{<:Real}, pF₀::Real, pH₀::AbstractVector, kH::AbstractVector, m::Real)
	T = eltype(p)
	pL₀, pU₀ = pH₀
	kL , kU  = kH

	max(pL₀+pU₀*exp(-2*kU*Z̄), pU₀+pL₀*exp(-2*kL*Z̄)) > 1 && return(T(NaN))

	iszero(length(μ)) && (μ = 0.)
	μ isa Number && (μ = fill(μ,length(τ)))

	mm1 = m - one(T)
	pD₀ = one(T) - pF₀
	pH = [m*pL₀*exp(kL*(kL/2-Z̄)), m*pU₀*exp(kU*(kU/2-Z̄))]

	LC = zeros(T, M.NC)  # likelihood for central/insignificant obs, left & right tails
	∫    = Vector{T}(undef, length(M.knots))  # pre-allocating this hampers automatic differentiation since type changes
	bufL = zeros(T, length(M.knots))
	bufU = zeros(T, length(M.knots))

	σ² = 1 .+ (τ² = τ.^2)
	𝒩  = Normal()
	𝒩μ = Normal.(μ, .√σ²)

	for (pᵢ,μᵢ,τᵢ²,σᵢ²,𝒩μᵢ) ∈ zip(p,μ,τ²,σ²,𝒩μ)
		# math on integration and interpolation points, outside loops
		Ω  = M.X * √(2τᵢ² / σᵢ²)  # 1st-order component of change of variables from pdf(Normal(ω)) to exp(-x²) for Gauss-Hermite quadrature
		ΩL = -Z̄ .- Ω
		ΩU =  Z̄ .- Ω
		𝒩Ω = Normal.(Ω)

		# lower tail
		kt1 = collect((μᵢ/τᵢ² .- M.knots) * -τᵢ²/σᵢ²)  #    -(z+μᵢ⁄τᵢ²)/(1+1⁄τᵢ²) for z at interpolation points -- negated 0th-order component of change of variables for quadrature
		kt2 = collect(kt1 - M.knots                 )  # z - (z+μᵢ⁄τᵢ²)/(1+1⁄τᵢ²) for z at interpolation points
		fill!(∫, zero(T))
		@inbounds Threads.@threads for j ∈ eachindex(M.knots)
			kt1j, kt2j = kt1[j], kt2[j]
			for (𝒩ω,ω,ωl,ωu,lnw) ∈ zip(𝒩Ω,Ω,ΩL,ΩU,M.lnW)  # quadrature integration
				∫[j] += exp(lnw - kL * (ω - kt1j) + logdiffcdf(𝒩, kt1j + ωu + kL, kt1j + ωl + kL) + mm1 * logccdf(𝒩ω, kt2j) - log1mexp(m * logccdf(𝒩, kt1j + ωl)))
			end
		end
		@. bufL += pᵢ * pdf(𝒩μᵢ, -M.knots) * (one(T) + pH[1] * ∫)

		# upper tail
		kt1 .= collect((μᵢ/τᵢ² .+ M.knots) * -τᵢ²/σᵢ²)  #    -(z+μᵢ⁄τᵢ²)/(1+1⁄τᵢ²) for z at interpolation points
		kt2 .= collect(kt1 + M.knots                 )  # z - (z+μᵢ⁄τᵢ²)/(1+1⁄τᵢ²) for z at interpolation points
		fill!(∫, zero(T))
		@inbounds Threads.@threads for j ∈ eachindex(M.knots)
			kt1j, kt2j = kt1[j], kt2[j]
			for (𝒩ω,ω,ωl,ωu,lnw) ∈ zip(𝒩Ω,Ω,ΩL,ΩU,M.lnW)  # quadrature integration
				∫[j] += exp(lnw + kU * (ω - kt1j) + logdiffcdf(𝒩, kt1j + ωu - kU, kt1j + ωl - kU) + mm1 * logcdf(𝒩ω, kt2j) - log1mexp(m * logcdf(𝒩, kt1j + ωu)))
			end
		end
		@. bufU += pᵢ * pdf(𝒩μᵢ, M.knots) * (one(T) + pH[2] * ∫)

		@. LC += pᵢ * pdf(𝒩μᵢ, M.zC)  # likelihoods for center/insignificant observations
	end

	@. bufL = log(bufL)
	@. bufU = log(bufU)
	llL = interpolate!(bufL, M.spline).(M.zLint)  # log likelihoods for lower tail
	llU = interpolate!(bufU, M.spline).(M.zUint)  # log likelihoods for upper tail

	ThreadsX.sum(llL) + ThreadsX.sum(llU) + ThreadsX.mapreduce(log, +, LC, init=zero(T)) +
		M.NC * log(pD₀) + mapreduce((Z̄pz,Z̄mz)->log1p(- pL₀ * exp(-kL * Z̄pz) - pU₀ * exp(-kU * Z̄mz)), +, M.Z̄pzC, M.Z̄mzC, init=zero(T)) - 
    xlog1py(M.N, -pF₀ * dot(p, @. diffcdf(𝒩μ,Z̄,-Z̄) - pL₀ * exp(kL*(σ²*kL/2-μ-Z̄)) * diffcdf(𝒩μ, Z̄+σ²*kL, -Z̄+σ²*kL) -
		                                                  pU₀ * exp(kU*(σ²*kU/2+μ-Z̄)) * diffcdf(𝒩μ, Z̄-σ²*kU, -Z̄-σ²*kU)))
end

# f(z|ω)
function fZcondΩ(z, ω; pF₀, pH₀, kH, m, truncate=true)
	pL₀,pU₀ = pH₀
	kL ,kU  = kH
	pD₀ = 1 - pF₀
	result = abs(z) < Z̄ ? pdf(Normal(ω),z) * pD₀ * (1 - pL₀ * exp(-kL*(Z̄+z)) - pU₀ * exp(-kU*(Z̄-z))) :
							          pdf(Normal(ω),z) + exp(z < 0 ? logpdf(MinNormal(m,ω),z) - logcdf( MinNormal(m,ω),-Z̄) + log(pL₀) + kL*(kL/2-Z̄-ω) + logdiffcdf(Normal(ω-kL),Z̄,-Z̄) :
											                                 logpdf(MaxNormal(m,ω),z) - logccdf(MaxNormal(m,ω), Z̄) + log(pU₀) + kU*(kU/2-Z̄+ω) + logdiffcdf(Normal(ω+kU),Z̄,-Z̄)  )
	truncate && (result /= (1 - pF₀ * (diffcdf(Normal(ω), Z̄,-Z̄) - pL₀ * exp(kL*(kL/2-Z̄-ω)) * diffcdf(Normal(ω-kL),Z̄,-Z̄) - 
	                                                              pU₀ * exp(kU*(kU/2-Z̄+ω)) * diffcdf(Normal(ω+kU),Z̄,-Z̄))))
	isnan(result) || isinf(result) ? 0. : result
end

# F(z|ω)
function FZcondΩ(z, ω; pF₀, pH₀, kH, m)
	pD₀ = 1 - pF₀
	pL₀,pU₀ = pH₀
	kL ,kU  = kH
	𝒩 = Normal(ω)
	D = diffcdf(Normal(ω), Z̄,-Z̄) - pL₀ * exp(kL*(kL/2-Z̄-ω)) * diffcdf(Normal(ω-kL),Z̄,-Z̄) - 
	                               pU₀ * exp(kU*(kU/2-Z̄+ω)) * diffcdf(Normal(ω+kU),Z̄,-Z̄)  # P[no p-hack]
	if z > Z̄  # tails
		𝒩max = MaxNormal(m,ω)
		result = 1 - (pU₀ * exp(logccdf(𝒩max,z) - logccdf(𝒩max, Z̄) + kU*(kU/2-Z̄+ω) + logdiffcdf(𝒩, Z̄-kU, -Z̄-kU)) + ccdf(𝒩,z)) / (1 - pF₀ * D)
	else
		if z < -Z̄
			𝒩min = MinNormal(m,ω)
			result =    pL₀ * exp(logcdf(𝒩min, z) - logcdf(𝒩min, -Z̄) + kL*(kL/2-Z̄-ω) + logdiffcdf(𝒩, kL+Z̄,  kL-Z̄)) + cdf(𝒩,z)
		else
			result =    pL₀ * exp(                                       kL*(kL/2-Z̄-ω) + logdiffcdf(𝒩, kL+Z̄, kL-Z̄)) + cdf(𝒩,-Z̄) + 
			                pD₀ * (diffcdf(Normal(ω), z,-Z̄) - pL₀ * exp(kL*(kL/2-Z̄-ω)) * diffcdf(Normal(ω-kL),z,-Z̄) - 
											                                  pU₀ * exp(kU*(kU/2-Z̄+ω)) * diffcdf(Normal(ω+kU),z,-Z̄)  )
		end
		result /= 1 - pF₀ * D
	end
	result
end

quantFcondΩ(q, ω; kwargs...) = find_zero(z -> q - FZcondΩ(z, ω; kwargs...), (-20,20))

# f(z), f(ω), f(ω|z), E[ω|z]
fZ = HnFl
fΩ(ω; p, μ, τ) = p'pdf.(Normal.(μ,τ), ω)
fΩcondZ(ω, z; p, μ, τ, kwargs...) = fZcondΩ(z, ω; kwargs..., truncate=false) * fΩ(ω; p, μ, τ) / fZ(z; p, μ, τ, kwargs..., truncate=false)
EΩcondZ(   z; p, μ, τ, kwargs...) = quadgk(ω -> ω * fΩcondZ(ω, z; p, μ, τ, kwargs...), -Inf, Inf)[1]

# CIs
Cquant(α, z; kwargs...) = find_zero(ω -> α - FZcondΩ(z, ω; kwargs...), (-20,20))
CI(    α, z; kwargs...) = Cquant(α/2, z; kwargs...), Cquant(1-α/2, z; kwargs...)


function HnFDGP(N; p::Vector, μ=0., τ::Vector, pF₀, pH₀, kH=[0.,0.], m=1, truncate=true, ω=NaN)
	μ isa Number && (μ = fill(μ,length(τ)))
	pL₀,pU₀ = pH₀
	kL ,kU  = kH

	if isnan(ω)
		I = rand(Categorical(p), N)
		Ω = map(i->rand(Normal(μ[i], τ[i])), I)
	else
		Ω = fill(ω,N)
	end

	Z✻ = rand.(Normal.(Ω))
	Z = similar(Z✻)
	@inbounds Threads.@threads for i ∈ eachindex(Z✻)
		Z✻ᵢ = Z✻[i]
		if abs(Z✻ᵢ) > Z̄
			Z[i] = Z✻ᵢ  # publish significant result as is
		else
			pL = pL₀ * exp(-kL*(Z̄+Z✻ᵢ))  # probability of hacking to lower tail
			pU = pU₀ * exp(-kU*(Z̄-Z✻ᵢ))  # probability of hacking to upper tail
			pF = pF₀ * (1 - pL - pU)
			pD = 1 - pL - pU - pF
			r = rand()
			if r < pD
				Z[i] = Z✻ᵢ  # publish insignificant result as is
			elseif r < pD + pF
				Z[i] = NaN  # file-drawer
			elseif r < pD + pF + pL
				𝒩 = MinNormal(m,Ω[i])
				Z[i] = quantile(𝒩, rand(Uniform(0., cdf(𝒩, -Z̄))))  # hack to lower tail
			else
				𝒩 = MaxNormal(m,Ω[i])
				Z[i] = quantile(𝒩, rand(Uniform(cdf(𝒩, Z̄), 1.)))  # hack to upper tail
			end
		end
	end

	if truncate
		keep = @. !isnan(Z) && abs(Z)<10
		Ω  = Ω[keep]
		Z✻ = Z✻[keep]
		Z  = Z[keep]
	end
	(Ω=Ω, Z✻=Z✻, Z=Z)  # named tuple with results
end

struct HnFResult<:RegressionModel
	depname::String
	coefnames::Vector{String}
	coef::Vector{Float64}
	vcov::Matrix{Float64}
	k::Int
	n::Int
	ll::Float64
	se::Vector{Float64}
	z::Vector{Float64}
	𝒩::Vector{Normal{Float64}}
	HnFResult(depname, coefnames, coef, vcov, k, n, ll) = (se = sqrtNaN.(diag(vcov)); new(depname, coefnames, coef, vcov, k, n, ll, se, coef ./ se, Normal.(coef,se)))
end

StatsAPI.aic( R::HnFResult) = 2 * (R.k − R.ll)
StatsAPI.aicc(R::HnFResult) = 2 * (R.k + R.k * (R.k − 1) / (R.n − R.k − 1) − R.ll)
StatsAPI.bic( R::HnFResult) = R.k * log(R.n) − 2 * R.ll
StatsAPI.coef(R::HnFResult) = R.coef
StatsAPI.coefnames(R::HnFResult) = R.coefnames
StatsAPI.confint(R::HnFResult; level::Real=0.95) = [quantile.(R.𝒩, (1-level)/2) cquantile.(R.𝒩, (1-level)/2)]
StatsAPI.coeftable(R::HnFResult; level::Real=0.95) = (CI = confint(R; level);
                                             CoefTable([R.coef, 
																						            R.se, 
																											  R.z,
																											  2ccdf.(Normal(), abs.(R.z)), 
																											  eachcol(CI)...],
																						           ["Estimate", "Std.Error", "z value", "Pr(>|z|)", "Lower 95%", "Upper 95%"],
																						           R.coefnames,
																						           4,
																						           3))
StatsAPI.dof(R::HnFResult) = R.k
# StatsAPI.informationmatrix(R::HnFResult; expected::Bool = true) = 
StatsAPI.isfitted(R::HnFResult) = true
StatsAPI.islinear(R::HnFResult) = false
# StatsAPI.loglikelihood(model::HnFResult, observation) = 
StatsAPI.loglikelihood(R::HnFResult) = R.ll
StatsAPI.nobs(R::HnFResult) = R.n
StatsAPI.vcov(R::HnFResult) = R.vcov
StatsAPI.weights(R::HnFResult) = UnitWeights(R.n)
StatsAPI.dof_residual(R::HnFResult) = R.n - R.k
StatsAPI.fitted(R::HnFResult) = R.coef
StatsAPI.responsename(R::HnFResult) = R.depname

function fit(M::HnFmodel, from::Union{AbstractDict,NamedTuple}; method::Optim.AbstractOptimizer=Newton())
	_from = from isa AbstractDict ? from : pairs(from)
	fromxform = [M.xforms[p](v) for (p,v) ∈ _from]

	# indexes to extract scalar and vector parameters from full parameter vector
	extractor = zip(keys(_from), Iterators.accumulate((ind,f)->f isa Number ? (last(ind)+1) : last(ind)+1:last(ind)+length(f), fromxform, init=0))

	objective(v) = -HnFll(M; (p => inverse(M.xforms[p])(v[e]) for (p,e) ∈ extractor)...)
	res = optimize(objective, vcat(fromxform...), method, autodiff=:forward)
	θ = Optim.minimizer(res)

	invxform = θ -> [θ[e] |> inverse(M.xforms[p]) for (p,e) ∈ extractor]
	b = NamedTuple([p=>θᵢ for ((p,_),θᵢ) ∈ zip(extractor,invxform(θ))])

	# use delta method to get se's for untransformed parameters
	Δ = ForwardDiff.jacobian(v->vcat(invxform(v)...), θ)
	H = ForwardDiff.hessian(objective, θ)
	Vxform = try pinv(H) catch _ fill(NaN, size(H)) end
	sexform = sqrtNaN.(diag(Vxform))
	V = Δ * Vxform * Δ'

	se = NamedTuple([p=> (Δᵢ = (e isa Int ? ForwardDiff.derivative : ForwardDiff.jacobian)(inverse(M.xforms[p]), θ[e]);
	                      Δᵢ isa Number ? sqrtNaN(Vxform[e,e])*abs(Δᵢ) : sqrtNaN.(diag(Δᵢ * Vxform[e,e] * Δᵢ')))
				           for (p,e) ∈ extractor])

	(res=res, Δ=Δ, H=H, Vxform=Vxform, sexform=sexform, V=V, se=se, b=b)
end

function fitnplot(z::Vector; D::Int=1, interpres=100, quadnodes=25, method::Optim.AbstractOptimizer=Newton(), from::NamedTuple=NamedTuple(), xform::NamedTuple=NamedTuple(),
									graphstub="", noplot::Bool=false, symmetric::Bool=false, zplot::StepRangeLen=(symmetric ? 0 : -5):.1:5, ωplot::StepRangeLen=(symmetric ? 0 : -5):.1:5, )

	from = merge((p=fill(1/D,D), μ=(symmetric ? Float64[] : fill(0.,D)), τ=collect(LinRange(1,D,D)), pF₀=.1, pH₀=[.1,.1], kH=[1., 1.], m=1.), from)
	xform = merge((p=SimplextoRⁿ, μ=identity, τ=bcast(log), pF₀=logit, pH₀=bcast(logit), kH=bcast(log), m=log), xform)

	M = HnFmodel(z; D, interpres, quadnodes, xform...)
	f = fit(M, from; method)

	one2D = string.(1:D)
	coefnames = vcat("p".* one2D, from.μ isa Number ? "μ"	: "μ".*one2D, "τ".* one2D, "pF₀", "pL₀", "pH₀", "kL", "kH", "m")
	b = vcat(f.b...)
	est = HnFResult(string(:z), coefnames, b, f.V, length(_b), size(z,1), -Optim.minimum(f.res))

	if !noplot
		t = NamedTuple([p=>(iszero(length(b)) && p==:μ ? 0. : b) for ((p,_),b) ∈ zip(pairs(from), f.b)])
		kwargsω = (p=t.p, μ=t.μ, τ=t.τ)
		kwargsz = (pF₀=t.pF₀, pH₀=t.pH₀, kH=t.kH, m=t.m)
		kwargsz0 = (pF₀=0, pH₀=[0.,0.], kH=[0.,0.], m=1)

		plt1 = stephist(z, normalize=:pdf, label="Actual published effects", legend=:topleft)  # outline histogram of data
		s,e = extrema(z); _zplot = s:.1:e
		pplottrue = map(z->t.p' * (@. pdf(Normal(kwargsω.μ,t.τ), z)), _zplot)
		pplotinitial = map(z->t.p' * (@. pdf(Normal(kwargsω.μ, √(t.τ^2+1)), z)), _zplot)
		pplotfit = map(z->HnFl(z; rtol=1e-2, kwargsω..., kwargsz...), _zplot)
		if symmetric
			pplottrue .*= 2
			pplotinitial .*= 2
			pplotfit .*= 2
		end
		plot!(_zplot, [pplottrue pplotinitial pplotfit], label=["Model: true effects" "Model: initial estimates" "Model: published estimates"], lw=[1 1 1])
		png("$graphstub fit")

		# distribution of z | ω=2
		ω = 2.
		plt2 = plot(zplot, mapreduce(z->[fZcondΩ(z, ω; kwargsz0...) fZcondΩ.(z, ω; kwargsz...)], vcat, zplot), label=["not distorted" "distorted"], xlabel="Reported z | true z = $ω", lw=[1 1])
		png("$graphstub z cond ω=$ω")
		
		# distribution of ω | z=2
		_z = 2.
		plt3 = plot(ωplot, mapreduce(ω->[fΩcondZ(ω,_z; kwargsω..., kwargsz0...) fΩcondZ(ω,_z; kwargsω..., kwargsz...)], vcat, ωplot), label=["not distorted" "distorted"], xlabel="True z | reported z = $_z", lw=[1 1])
		png("$graphstub ω cond z=$_z")
		
		# frequentist CI's as fn of z
		plt4 = plot(ωplot, mapreduce(ω->[Cquant.([.025 .5 .975], ω; kwargsz0...)..., Cquant.([.025 .5 .975], ω; kwargsz...)...]',vcat,ωplot), linecolor=[:blue :blue :blue :orange :orange :orange], lw=[1 1 1 1 1 1], linestyle=[:solid :dash :solid :solid :dash :solid], legend=false, xlabel="Reported z", ylabel="95% CI & median")
		png("$graphstub CI cond z")
		
		# Bayesian posterior mean of ω as fn of Z
		pplot = mapreduce(z->[z EΩcondZ(z; kwargsω..., kwargsz0...) EΩcondZ(z; kwargsω..., kwargsz...)], vcat, zplot)
		plt5 = plot(zplot, pplot, label=["As is" "shrinkage from informative prior" "shrinkage + adjustment for distortion"], xlabel="Reported z", ylabel="Expected true z", lw=[1 1 1])
		png("$graphstub E[ω] cond z")
		
		# E[ω] discount
		plt6 = plot(zplot[zplot.>.2], pplot[zplot.>.2,2:3]./zplot[zplot.>.2], label=["shrinkage from informative prior" "shrinkage + adjustment for distortion"], xlabel="Reported z", ylabel="Discount multiplier", lw=[1 1 1])
		png("$graphstub E[ω] discount")

		plot(plt1, plt2, plt3, plt4, plt5, plt6, size=(2700,1950), dpi=300)
		png("$graphstub all")

		pplot = vcat(map(ω -> [0. quantFcondΩ(.5, ω; kwargsz...) - ω], ωplot)...)
		plt7 = plot(ωplot, pplot)
		Cplot = vcat(map(ω ->[.95 FZcondΩ(ω+Z̄, ω; kwargsz...)-FZcondΩ(ω-Z̄, ω; kwargsz...)], ωplot)...)
		plt8 = plot(ωplot, Cplot)
		plot(plt7, plt8, title="Andrews & Kasy (2019) Figure 1")
		png("$graphstub A&K Fig1")
	end
	est
end


# confirm match between model and simulation
p = [1.]
μ = [0.7]
τ = [2.]
pF₀ = .3
pH₀ = [.2,.3]
kH = [1.,.5]
m = 5.
kwargs = (p=p, μ=μ, τ=τ, pF₀=pF₀, pH₀=pH₀, kH=kH, m=m)

Random.seed!(1231)
z = HnFDGP(3_000_000; kwargs..., truncate=true).Z

histogram(z, normalize=:pdf)
zplot = -10:.1:10
pplot = map(z->HnFl(z; kwargs..., truncate=true), zplot)
plot!(zplot, pplot)

M = HnFmodel(z, D=length(τ), p=SimplextoRⁿ, μ=identity, τ=bcast(log), pF₀=logit, pH₀=bcast(logit), kH=bcast(log), m=log)
@time f = fit(M, kwargs)
plot!(zplot, map(z->HnFl(z; f.b...), zplot))


nostar(coef, p) = coef

@time begin
	# Georgescu and Wren 2018 ~1M sample, doi:10.1093/bioinformatics/btx811, github.com/agbarnett/intervals
	df = DataFrame(CSV.File(raw"D:\OneDrive\Documents\Work\Clients & prospects\GiveWell\Noisy data\Georgescu.Wren.csv"))
	@. df.cilevel[ismissing(df.cilevel) || df.cilevel==.0095 || df.cilevel==.05] = .95
	@. df.z = log(df.mean) / (ifelse(ismissing(df.lower) || iszero(df.lower), log(df.upper / df.mean), log(df.upper / df.lower) / 2) / cquantile(Normal(), (1 - df.cilevel)/2.))
	@. @subset!(df, !ismissing(:z) && !ismissing(:lower))
	@. @subset!(df, iszero(:mistake) && abs(:z) < 10.)  # van Zwet & Cator Figure 1 stops at 10
	@. @subset!(df, :source!="Abstract")
	df.z = Float64.(df.z)

	@time fGW1 = fitnplot(df.z; graphstub="Georgescu-Wren")
	@time fGW2 = fitnplot(df.z; D=2, graphstub="Georgescu-Wren 2", method=LBFGS())
	regtable(fGW1, fGW2; estim_decoration=nostar, 
	                     regression_statistics=[:nobs],
											 custom_statistics=(ll=loglikelihood.([fGW1, fGW2]),),
											 print_estimator_section=false,
											 regressors = ["p1", "p2", "μ1", "μ2", "τ1", "τ2", "pF₀", "pL₀", "pU₀", "kL", "kU", "m"],
											 estimformat="%0.2g",
											 statisticformat="%0.2g")

	# println("Number of missing studies = ", size(df,1) * p̂F * p̂'*(@. cdf(Normal(μ,√(1+τ^2)),Z̄)-cdf(Normal(μ,√(1+τ^2)),-Z̄)))
	# println("Number of p-hacked studies = ", size(df,1) * (1-p̂D-p̂F) * p̂'*(@. cdf(Normal(μ,√(1+τ^2)),Z̄)-cdf(Normal(μ,√(1+τ^2)),-Z̄)))

	# van Zwet, Schwab, and Senn 2021 data, osf.io/xq4b2
	df = DataFrame(CSV.File(raw"D:\OneDrive\Documents\Work\Clients & prospects\GiveWell\Noisy data\CochraneEffects.csv"))
	@. @subset!(df, abs(:z) < 20 && :"outcome.nr"!=1 && :RCT=="yes" && :"outcome.group"=="efficacy")
	combine(groupby(df, :"study.id.sha1"), :z => sample => :z)  # randomly choose among primary outcomes
	df.z .*= rand([-1. 1.], nrow(df))  # symmetrize data without duplication

	@time fvZSS = fitnplot(df.z; D=4, from=(p=[.32,.31,.3,.07], μ=0., τ=[.61, 1.42, 2.16, 5.64], pF₀=.01, pH₀=[.01, .01], kH=[0.,0.], m=3.), graphstub="vZSS")

	# Star Wars, doi.org/10.1257/app.20150044, openicpsr.org/openicpsr/project/113633/version/V1/view?path=/openicpsr/113633/fcr:versions/V1/brodeur_le_sangnier_zylberberg_replication/Data/Final/final_stars_supp.dta&type=file
	df = DataFrame(CSV.File(raw"D:\OneDrive\Documents\Work\Library\Meta-science\Brodeur et al. 2016\Data\Final\final_stars_supp.csv"))
	@. @subset!(df, lowercase(:main)=="yes")
	df = DataFrame(Dict(:z => df.coefficient_num ./ df.standard_deviation_num))
	dropmissing!(df)
	@subset!(df, abs.(:z).<10)  # Star Wars graphs stop at 10
	df.z = Float64.(df.z)

	fSW3 = fitnplot(df.z; D=3, xform=(kH=identity,), noplot=true)  # hack: 3-component model converging to higher-likelihood 2-component solution, maybe because of greater flexibility?
	# keep = abs.(fSW3.b.p) .> 1e-3
	# fSW2 = fitnplot(df.z; D=2, xform=(kH=identity,), from=NamedTuple(k=>v isa Vector ? v[keep] : v for (k,v) ∈ pairs(fSW3.b)), graphstub="Star Wars 3")
	# fSW4 = fitnplot(df.z; D=4, xform=(kH=identity,), graphstub="Star Wars 4")

	fSW3sym = fitnplot(abs.(df.z); D=3, xform=(pH₀=logit ∘ consvec[2], kH=logit ∘ consvec[2]), symmetric=true, graphstub="fSW3sym")  # symmetrized model

	# Arel-Bundock et al. histogram
	img = load("Arel-Bundock et al. histogram.png")
	hist = [round(Int, 383*(1-(findfirst(<(.5), channelview(img)[1,:,round(Int,c)])-15)/(2049-15))) for c ∈ range(13, 3180; length=200)]  # 😄
	bar(.025:.05:10, hist)
	z = vcat([fill(z,h) for (z,h) ∈ zip(.025:.05:10, hist)]...)
	f1 = fitnplot(z; D=1, xform=(pH₀=logit ∘ consvec[2], kH=logit ∘ consvec[2]), symmetric=true, graphstub="A-B et al. 1")
	f2 = fitnplot(z; D=2, xform=(pH₀=logit ∘ consvec[2], kH=logit ∘ consvec[2]), symmetric=true, graphstub="A-B et al. 2")
	f3 = fitnplot(z; D=3, xform=(pH₀=logit ∘ consvec[2], kH=logit ∘ consvec[2]), symmetric=true, graphstub="A-B et al. 3")

	# Brodeur, Cook, and Heyes 2020, DOI 10.1257/aer.20190687, openicpsr.org/openicpsr/project/120246/version/V1/view?path=/openicpsr/120246/fcr:versions/V1/MM-Data.dta&type=file
	df = DataFrame(CSV.File(raw"D:\OneDrive\Documents\Work\Library\Meta-science\Brodeur, Cook, and Heyes 2020\MM Data.csv"))
	rename!(df, :t => :abst)
	df.t = df.mu ./ df.sd
	@. @subset!(df, abs(:t)<10)
	df.t = Float64.(df.t)
	df.abst = Float64.(df.abst)
	fBCH1 = fitnplot(df.abst; D=1, xform=(pH₀=logit ∘ consvec[2], kH=logit ∘ consvec[2]), symmetric=true, graphstub="BCH 1")
	fBCH2 = fitnplot(df.abst; D=2, xform=(pH₀=logit ∘ consvec[2], kH=logit ∘ consvec[2]), symmetric=true, graphstub="BCH 2")
	fBCH3 = fitnplot(df.abst; D=3, xform=(pH₀=logit ∘ consvec[2], kH=logit ∘ consvec[2]), symmetric=true, graphstub="BCH 3")

	# Vivalt 2020
	df = DataFrame(CSV.File(raw"D:\OneDrive\Documents\Work\Library\Meta-science\Vivalt 2020\Data\data_unstandardized.csv"))
	df.z = df.treatmentcoefficient ./ df.treatmentstandarderror
	@. @subset!(df, abs(:z)<10)
	fV1 = fitnplot(df.z; D=1, graphstub="V 1")
	fV2 = fitnplot(df.z; D=2, graphstub="V 2")
end
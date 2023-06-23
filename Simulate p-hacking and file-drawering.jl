using Random, Distributions, Interpolations, Base.Iterators, FastGaussQuadrature, BenchmarkTools, Optim, LogExpFunctions, Plots, CSV, DataFrames, DataFramesMeta, ForwardDiff, LinearAlgebra, Roots, QuadGK, Statistics, ThreadsX

const Z̄ = 1.9599639845401

# fold the unit-square parameter space across a+b=1
fold(a,b) = a+b>1 ? (1-b,1-a) : (a,b)

@inline diffcdf(N,a,b) = cdf(N,b) - cdf(N,a)

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
function RⁿtoSimplex(q::Vector{T}) where T
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
function SimplextoRⁿ(p::Vector{T}) where T
	q = Vector{T}(undef, length(p)-1)
	sum = p[end]
	for i ∈ reverse(eachindex(q))
		sum += p[i]
		q[i] = acos(√(p[i] / sum)) / π
	end
	q .= logit.(q)
end


function HnFl(z::T, p::Vector{T}, μ::Vector, τ::Vector{T}, pD::T, pF::T, u::T, m::T; truncate=true) where T<:Number
	pH = m * (1 - pD - pF)
	length(p) < length(μ) && (p = [p; 1-sum(p)])
	𝒩  = Normal()
	𝒩μ = @. Normal(μ, √(1 + τ^2))
	𝒩ω = @. NormalCanon(z + μ/τ^2, 1 + 1/τ^2)
	if abs(z) ≥ Z̄
		result = zero(T)
		@inbounds for (pᵢ,𝒩μᵢ,𝒩ωᵢ) ∈ zip(p,𝒩μ,𝒩ω)
			if z < 0
				result += pᵢ * pdf(𝒩μᵢ, z) * (one(T) + pH * (1-u) * quadgk(ω -> (a = pdf(𝒩ωᵢ,ω)*(cdf(𝒩,Z̄-ω)-cdf(𝒩,-Z̄-ω)) * ccdf(𝒩,z-ω)^(m-1) / (1-ccdf(𝒩, -Z̄-ω)^m);
																																		 isnan(a) || isinf(a) ? 0. : a), 
																                               -Inf, Inf)[1])
			else
				result += pᵢ * pdf(𝒩μᵢ, z) * (one(T) + pH *    u  * quadgk(ω -> (a = pdf(𝒩ωᵢ,ω)*(cdf(𝒩,Z̄-ω)-cdf(𝒩,-Z̄-ω)) * cdf(𝒩,z-ω)^(m-1) / (1 - cdf(𝒩,  Z̄-ω)^m);
																																		 isnan(a) || isinf(a) ? 0. : a), 
																                               -Inf, Inf)[1])
			end
		end
	else
		result = pD * dot(p, pdf.(𝒩μ, z))
	end
	truncate && (result /= (1 - pF * dot(p, (cdf.(𝒩μ,Z̄) - cdf.(𝒩μ,-Z̄)))))
	result
end

# object to hold pre-computed stuff for log likelihood computation
struct HnFstuff
	D::Int  # number of mixture components
	z::Vector{Float64}  # all data
	zC::Vector{Float64}  # just the central, insignificant z's
	N::Int; NC::Int; NL::Int; NU::Int  # number of z's, insigficant z's, lower significant z's, upper
	knots::LinRange  # interpolation knots in [Z̄,max]
	spline::Interpolations.InterpolationType  # type of interpolation
	zSint::NTuple{2,Vector{Float64}}  # lower- & upper-tail significant z values mapped to cardinal knot numbering space since interpolate() is faster with cardinally spaced knots
	X::Vector{Float64}; W::Vector{Float64}  # quadrature nodes & weights
	lnW::Vector{Float64}
end
# constructor
function HnFstuff(z::Vector{Float64}; D::Int, interpres::Int, quadnodes::Int)
	zC = z[abs.(z) .< Z̄]

	s = Z̄ - 3/interpres; e = maximum(z)+.1
	knots = s : 1/interpres : e  # LinRange(s, e, ceil(Int, (e - s) * interpres) + 1)
	zSint = (-z[z .≤ -Z̄] .- s) .* interpres .+ 1, (z[z .≥ Z̄] .- s) .* interpres .+ 1  # map tail z values to knot numbering 1, 2, ... for Z̄-3/interpres, Z̄-2/interpres, ...
	
	X, W = gausshermite(quadnodes)
	W ./= √π
	
	HnFstuff(D, z, zC, length(z), length(zC), length.(zSint)..., knots, BSpline(Quadratic(Free(OnGrid()))), zSint, X, W, log.(W))
end

# bulk log probabilities as function of data & parameters, for estimation
function HnFll(o::HnFstuff, p::Vector{T}, μ::Vector, τ::Vector{T}, pD::T, pF::T, u::T, m::T) where T<:Number
	τ = exp.(τ)
	m = exp(m); mm1 = m - one(T)
	u = logistic(u)
	p = RⁿtoSimplex(p)
	pD, pF, _pH = RⁿtoSimplex([pD,pF])
	pH = m * _pH * [one(T)-u u]

	LC = fill(zero(T), o.NC)  # likelihood for insignificant obs
	LS = fill(zero(T), o.NL), fill(zero(T), o.NU)  # for significant obs, left & right tails

	𝒩  = Normal()
	𝒩μ = Normal.(μ, .√(1 .+ τ.^2))

	for (pᵢ,μᵢ,τᵢ,𝒩μᵢ) ∈ zip(p,μ,τ,𝒩μ)
		σ² = τᵢ^2; σ² /= one(T)+σ²; σ = √σ²

		# math on integration and interpolation points, outside loops
		X  = o.X * (√2 * σ)
		XU =  Z̄ .- X
		XL = -Z̄ .- X
		𝒩X = Normal.(X)

		buf = Vector{T}(undef, length(o.knots))  # pre-allocating this hampers automatic differentiation since type changes

		# lower tail
		k1 = collect((μᵢ/τᵢ^2 .- o.knots) .* -σ²)  #    -(z+μᵢ⁄τᵢ²)/(1+1⁄τᵢ²) for z at interpolation points
		k2 = collect(k1 - o.knots  )               # z - (z+μᵢ⁄τᵢ²)/(1+1⁄τᵢ²) for z at interpolation points
		fill!(buf, zero(T))
		@inbounds Threads.@threads for j ∈ eachindex(o.knots)
			k1j, k2j = k1[j], k2[j]
			for (𝒩x,xl,xu,lnw) ∈ zip(𝒩X,XL,XU,o.lnW)  # quadrature integration
				buf[j] += exp(lnw + logdiffcdf(𝒩, k1j + xu, k1j + xl) + mm1 * logccdf(𝒩x, k2j) - log1mexp(m * logccdf(𝒩, k1j + xl)))
			end
		end
		buf .= pᵢ .* pdf.(𝒩μᵢ, -o.knots) .* (one(T) .+ pH[1] .* buf)
		LS[1] .+= interpolate!(buf, o.spline).(o.zSint[1])  # likelihoods for significant observations

		# upper tail
		k1 .= collect((μᵢ/τᵢ^2 .+ o.knots) .* -σ²)  #    -(z+μᵢ⁄τᵢ²)/(1+1⁄τᵢ²) for z at interpolation points
		k2 .= collect(k1 + o.knots               )  # z - (z+μᵢ⁄τᵢ²)/(1+1⁄τᵢ²) for z at interpolation points
		fill!(buf, zero(T))
		@inbounds Threads.@threads for j ∈ eachindex(o.knots)
			k1j, k2j = k1[j], k2[j]
			for (𝒩x,xl,xu,lnw) ∈ zip(𝒩X,XL,XU,o.lnW)  # quadrature integration
				buf[j] += exp(lnw + logdiffcdf(𝒩, k1j + xu, k1j + xl) + mm1 * logcdf(𝒩x, k2j) - log1mexp(m * logcdf(𝒩, k1j + xu)))
			end
		end
		buf .= pᵢ .* pdf.(𝒩μᵢ, o.knots) .* (one(T) .+ pH[2] .* buf)
		LS[2] .+= interpolate!(buf, o.spline).(o.zSint[2])  # likelihoods for significant observations

		LC .+= (pᵢ * pD) .* pdf.(𝒩μᵢ, o.zC)  # likelihoods for center/insignificant observations
	end
	return(mapreduce(v->ThreadsX.mapreduce(log, +, v, init=zero(T)), +, (LC,LS...)) - xlog1py(o.N, -pF * dot(p, (cdf.(𝒩μ,Z̄) - cdf.(𝒩μ,-Z̄)))))
end

# log likelihood--function of parameters only
negHnFll(o)        = v -> -HnFll(o, v[1:o.D-1], v[o.D:2*o.D-1], v[2*o.D:3*o.D-1], v[3*o.D:end]...)
negHnFll0cent(o)   = v -> -HnFll(o, v[1:o.D-1], zeros(Float64,o.D), v[o.D:2*o.D-1], v[2*o.D:end]...)  # impose μ=0
negHnFllSharedμ(o) = v -> -HnFll(o, v[1:o.D-1], fill(v[o.D],o.D), v[o.D+1:2*o.D], v[2*o.D+1:end]...) # impose shared μ

function HnFCDF(o::HnFstuff, z::T, p::Vector{T}, μ::Vector{T}, τ::Vector{T}, pD::T, pF::T, U) where T<:Number
	pH = 1 - pD - pF

	𝒩H = Normal.(√2τ * o.X' .+ μ)
	𝒩  = Normal.(μ, .√(1 .+ τ.^2))

	a = cdf.(𝒩H, -Z̄)
	b = ccdf.(𝒩H, Z̄)
	if z ≤ Z̄  # tails
		result = pH * p'* cdf.(𝒩, min(z,-Z̄)                          ) + pH * (p' * ( cdf.(𝒩H, min(z,-Z̄))       .* (one(T) .- a .- b) ./ (a .+ U .* b)) * o.W)
	else
		result = pH * p'*(cdf.(𝒩,       -Z̄) + cdf.(𝒩,z) - cdf.(𝒩,Z̄)) + pH * (p' * ((a + U * (cdf.(𝒩H,z) - b)) .* (one(T) .- a .- b) ./ (a .+ U .* b)) * o.W)
	end
	z > -Z̄ &&  # central bit
		(result += pD * (p' * (cdf.(𝒩, min(z,Z̄)) - cdf.(𝒩,-Z̄))))
	result / (1 - pF * (p' * (cdf.(𝒩,Z̄) - cdf.(𝒩,-Z̄))))
end

# f(z|ω)
function fZcondΩ(z, ω, pD, pF, u, m; truncate=true)
	pH = 1 - pD - pF
	result = abs(z) < Z̄ ? pD * pdf(Normal(ω),z) :
	         pdf(Normal(ω),z) + pH * exp(logdiffcdf(Normal(ω),Z̄,-Z̄) + (z ≤ -Z̄ ? log1p(-u) + logpdf(MinNormal(m,ω),z) -  logcdf(MinNormal(m,ω),-Z̄) :
					                                                                    log(   u) + logpdf(MaxNormal(m,ω),z) - logccdf(MaxNormal(m,ω), Z̄)  ))
	truncate && (result /= (1 - pF *(cdf(Normal(ω), Z̄) - cdf(Normal(ω),-Z̄))))
	isnan(result) || isinf(result) ? 0. : result
end

# F(z|ω)
function FZcondΩ(z, ω, pD, pF, u, m)
	pH = 1 - pD - pF
	l = 1 - u
	𝒩 = Normal(ω)
	a = cdf(𝒩,-Z̄)
	b = cdf(𝒩, Z̄)
	if z > Z̄  # tails
		𝒩max = MaxNormal(m,ω)
		result = 1 - (u * pH * exp(logccdf(𝒩max,z) - logccdf(𝒩max,Z̄) + logdiffcdf(𝒩, Z̄, -Z̄)) + ccdf(𝒩,z)) / (1-pF*(b-a))
	else
		𝒩min = MinNormal(m,ω)
		if z < -Z̄
			result = l * pH * exp(logcdf(𝒩min, z) - cdf(𝒩min, -Z̄) + logdiffcdf(𝒩, Z̄, -Z̄)) + cdf(𝒩,z)
		else
			result = l * pH * b + (1 - l * pH - pD) * a + pD * cdf(𝒩,z)
		end
		result /= 1-pF*(b-a)
	end
	result
end


# f(z), f(ω), f(ω|z), E[ω|z]
fZ = HnFl
fΩ(ω, p, μ, τ) = p'pdf.(Normal.(μ,τ), ω)
fΩcondZ(ω, z, p, μ, τ, pD, pF, u, m) = fZcondΩ(z, ω, pD, pF, u, m, truncate=false) * fΩ(ω, p, μ, τ) / fZ(z, p, μ, τ, pD, pF, u, m, truncate=false)
EΩcondZ(z, p, μ, τ, pD, pF, u, m) = quadgk(ω->ω * fΩcondZ(ω, z, p, μ, τ, pD, pF, u, m), -Inf, Inf)[1]

# CIs
Cquant(α, z, pD, pF, u, m) = find_zero(ω -> α - FZcondΩ(z, ω, pD, pF, u, m), (-20,20))
CI(α, z, pD, pF, u, m) = Cquant(α/2, z, pD, pF, u, m), Cquant(1-α/2, z, pD, pF, u, m)


function HnFDGP(N, p, μ, τ, pD, pF, u, m)
	p = vcat(p)
	μ = vcat(μ)
	τ = vcat(τ)
	length(p) < length(μ) && (p = [p; 1-sum(p)])
	pDF = pD + pF
	pDFu = pDF + (1 - pDF) * u

	I = rand(Categorical(p), N)
	Ω = map(i->rand(Normal(μ[i], τ[i])), I)
	Z✻ = rand.(Normal.(Ω))
	Z = similar(Z✻)
	@inbounds Threads.@threads for i ∈ eachindex(Z✻)
		if abs(Z✻[i]) > Z̄
			Z[i] = Z✻[i]
		else
			r = rand()
			if r < pD
				Z[i] = Z✻[i]
			elseif r < pDF
				Z[i] = NaN
			elseif r < pDFu
				𝒩 = MaxNormal(m,Ω[i])
				Z[i] = quantile(𝒩, rand(Uniform(cdf(𝒩, Z̄), 1.)))
			else
				𝒩 = MinNormal(m,Ω[i])
				Z[i] = quantile(𝒩, rand(Uniform(0., cdf(𝒩, -Z̄))))
			end
		end
	end
	keep = .!isnan.(Z) .&& abs.(Z).<10
	(Ω=Ω[keep], Z✻=Z✻[keep], Z=Z[keep])
end


# confirm match between model and simulation
pD = .4
pF = .3
u = .8
p = [.7]
μ = [-1.,1.]
τ = [2.,2.]
m = 5.6

z = HnFDGP(3_000_000,p,μ,τ,pD,pF,u,m).Z

histogram(z, normalize=:pdf, legend=false)
zplot = -9:.1:9
plot!(zplot, map(z->.01+HnFl(z,p,μ,τ,pD,pF,u,m), zplot))
plot!(zplot, map(z->.02+exp(-negHnFll(HnFstuff([z], D=length(μ), interpres=300, quadnodes=25))(vcat(logit.(p),μ,log.(τ),logit(pD),logit(pF),logit(u),log(m)))),zplot))

o = HnFstuff(z, D=length(μ), interpres=300, quadnodes=25)
@time res = optimize(negHnFll(o), vcat(logit.(p),μ,log.(τ),logit(pD),logit(pF),logit(u),log(m)), LBFGS(), autodiff=:forward)
θ₂ = Optim.minimizer(res)
p̂, μ̂ , τ̂ , p̂D, p̂F, û, m̂ = logistic.(θ₂[1:o.D-1]), θ₂[o.D:2*o.D-1], exp.(θ₂[2*o.D:3*o.D-1]), logistic(θ₂[3*o.D]), logistic(θ₂[3*o.D+1]), logistic(θ₂[3*o.D+2]), exp(θ₂[3*o.D+3])
p̂ = vcat(p̂, 1-sum(p̂))
p̂D, p̂F = fold(p̂D, p̂F)
println((p̂=p̂, μ̂ =μ̂ , τ̂ =τ̂ , p̂D=p̂D, p̂F=p̂F, û=û, m̂=m̂))

plot!(zplot, map(z->.03+HnFl(z, p̂, μ̂ , τ̂ , p̂D, p̂F, û, m̂), zplot))


# data prep
df = DataFrame(CSV.File(raw"D:\OneDrive\Documents\Work\Clients & prospects\GiveWell\Noisy data\Georgescu.Wren.csv"))
@. df.cilevel[ismissing(df.cilevel) || df.cilevel==.0095 || df.cilevel==.05] = .95
@. df.z = log(df.mean) / (ifelse(ismissing(df.lower) || iszero(df.lower), log(df.upper / df.mean), log(df.upper / df.lower) / 2) / cquantile(Normal(), (1 - df.cilevel)/2.))
@. @subset!(df, !ismissing(:z) && !ismissing(:lower))
@. @subset!(df, iszero(:mistake) && abs(:z) < 10.)
@. @subset!(df, :source!="Abstract")
df.z = Float64.(df.z)
histogram(df.z, normalize=:pdf, label="Actual published effects", legend=:topleft)

# fit 2-component min/max model
p = Float64[.5,.5]
μ = [0.,0.]
τ = [1.,2.]
pD = .4
pF = .5
pH = 1 - pD - pF
u = .75
m = 3.

o = HnFstuff(df.z, D=length(μ), interpres=300, quadnodes=25)
@time res2 = optimize(negHnFll(o), vcat(SimplextoRⁿ(p),μ,log.(τ),SimplextoRⁿ([pD,pF,pH])...,logit(u),log(m)), LBFGS(), autodiff=:forward)
θ, ll = Optim.minimizer(res2), Optim.minimum(res2)
p̂, μ̂ , τ̂ , p̂D, p̂F, p̂H, û, m̂ = RⁿtoSimplex(θ[1:o.D-1]), θ[o.D:2*o.D-1], exp.(θ[2*o.D:3*o.D-1]), RⁿtoSimplex(θ[3*o.D:3*o.D+1])..., logistic(θ[3*o.D+2]), exp(θ[3*o.D+3])
println((p̂=p̂, μ̂ =μ̂ , τ̂ =τ̂ , p̂D=p̂D, p̂F=p̂F, p̂H=p̂H, û=û, m̂=m̂))
println("Mean ω = ", p̂'μ̂)
ses2 = sqrt.(diag(pinv(ForwardDiff.hessian(negHnFll(o), θ₂))))

zplot = -10:.1:10
pplotfit = map(z->HnFl(z, p̂, μ̂ , τ̂ , p̂D, p̂F, û, m̂), zplot)
plt = histogram(df.z, normalize=:pdf, label="Actual published estimates", legend=:topleft)
plot!(zplot, [p̂[1] * pdf.(Normal(μ̂[1],τ̂[1]), zplot)+p̂[2] * pdf.(Normal(μ̂[2],τ̂[2]), zplot) p̂[1] * pdf.(Normal(μ̂[1],√(1+τ̂[1]^2)), zplot)+p̂[2] * pdf.(Normal(μ̂[2],√(1+τ̂[2]^2)), zplot) pplotfit], label=["Model: true effects" "Model: initial estimates" "Model: published estimates"])
png(plt, "fit")
plt

# distribution of z | ω=2
ω=2; zplot=-3:.01:5; plot(zplot, mapreduce(z->fZcondΩ.(z, ω, [1 p̂D], [0 p̂F], [1 û], [1 m̂]), vcat, zplot), label=["not distorted" "distorted"], xlabel="Reported z | true z = 2")
png("z cond ω=2")

# distribution of ω | z=2
z=2.; ωplot=-3:.01:5; plot(ωplot, mapreduce(ω->[fΩcondZ(ω,z,p̂,μ̂ ,τ̂ ,1.,0.,1.,1.) fΩcondZ(ω,z,p̂,μ̂ ,τ̂ ,p̂D,p̂F,û,m̂)], vcat, ωplot), label=["not distorted" "distorted"], xlabel="True z | reported z = 2")
png("ω cond z=2")

# frequentist CI's as fn of z
ωplot=-5:.01:5; plot(ωplot, ThreadsX.mapreduce(ω->[Cquant.([.025 .5 .975], ω, 1, 0, 1, 1)..., Cquant.([.025 .5 .975], ω, p̂D, p̂F, û, m̂)...]',vcat,ωplot), linecolor=[:blue :blue :blue :orange :orange :orange], linestyle=[:solid :dash :solid :solid :dash :solid], legend=false, xlabel="Reported z", ylabel="95% CI & median")
png("CI cond z")

# Bayesian posterior mean of ω as fn of Z
zplot=-5.:.1:5
pplot = mapreduce(z->[z EΩcondZ(z,p̂,μ̂ ,τ̂ ,1.,0.,1.,1.) EΩcondZ(z,p̂,μ̂ ,τ̂ ,p̂D,p̂F,û, m̂)], vcat, zplot)
plot(zplot, pplot, label=["As is" "shrinkage from informative prior" "shrinkage + adjustment for distortion"], xlabel="Reported z", ylabel="Expected true z")
png("E[ω] cond z")

plot(zplot[zplot.>.2], pplot[zplot.>.2,2:3]./zplot[zplot.>.2], label=["shrinkage from informative prior" "shrinkage + adjustment for distortion"], xlabel="Reported z", ylabel="Discount multiplier")
png("E[ω] discount")

println("Number of missing studies = ", size(df,1) * p̂F * p̂'*(@. cdf(Normal(μ,√(1+τ^2)),Z̄)-cdf(Normal(μ,√(1+τ^2)),-Z̄)))
println("Number of p-hacked studies = ", size(df,1) * (1-p̂D-p̂F) * p̂'*(@. cdf(Normal(μ,√(1+τ^2)),Z̄)-cdf(Normal(μ,√(1+τ^2)),-Z̄)))

# CDF of z conditional on fitted parameters
p = vcat([[z  p̂'cdf.(Normal.(μ̂ , .√(τ̂ .^2 .+ 1)),z) HnFCDF(o,z,p̂,μ̂ ,τ̂ ,p̂D,p̂F,û, m̂)] for z ∈ LinRange(-10:.1:10)]...)
plot(p[:,1], p[:,2:3], label=["Unfiltered" "Filtered"])


# van Zwet, Schwab, and Senn 2021 data, osf.io/xq4b2
df = DataFrame(CSV.File(raw"D:\OneDrive\Documents\Work\Clients & prospects\GiveWell\Noisy data\CochraneEffects.csv"))
@. @subset!(df, abs(:z) < 20 && :"outcome.nr"!=1 && :RCT=="yes" && :"outcome.group"=="efficacy")
# combine(groupby(df, :"study.id.sha1"), :z => sample => :z)  # randomly choose among primary outcomes
df.z .*= rand([-1. 1.], nrow(df))
histogram(df.z, normalize=:pdf, label="Actual published effects", legend=:topleft)

p = Float64[.3,.3,.3,.1]
μ = [0.,0.,0.,0.]
τ = [.61, 1.42, 2.16, 5.64]  # vZSS Table 1
pD = .99
pF = .01
pH = 1 - pD - pF
u = .75
m = 3.

o = HnFstuff(df.z, D=length(μ), interpres=300, quadnodes=25)
@time res = optimize(negHnFll(o), vcat(SimplextoRⁿ(p),μ,log.(τ),SimplextoRⁿ([pD,pF,pH])...,logit(u),log(m)), LBFGS(), autodiff=:forward)
θ, ll = Optim.minimizer(res), Optim.minimum(res)
p̂, μ̂ , τ̂ , p̂D, p̂F, p̂H, û, m̂ = RⁿtoSimplex(θ[1:o.D-1]), θ[o.D:2*o.D-1], exp.(θ[2*o.D:3*o.D-1]), RⁿtoSimplex(θ[3*o.D:3*o.D+1])..., logistic(θ[3*o.D+2]), exp(θ[3*o.D+3])
println((p̂=p̂, μ̂ =μ̂ , τ̂ =τ̂ , p̂D=p̂D, p̂F=p̂F, p̂H=p̂H, û=û, m̂=m̂))
println("Mean ω = ", p̂'μ̂)

ses = sqrt.(diag(pinv(ForwardDiff.hessian(negHnFll(o), θ))))

# Star Wars
df = DataFrame(CSV.File(raw"D:\OneDrive\Documents\Work\Library\Meta-science\Brodeur et al. 2016\Data\Final\final_stars_supp.csv"))
@. @subset!(df, lowercase(:main)=="yes")
df = DataFrame(Dict(:z => df.coefficient_num ./ df.standard_deviation_num))
dropmissing!(df)
@subset!(df, abs.(:z).<20)
df.z = Float64.(df.z)
histogram(abs.(df.z), normalize=:pdf, bins=1000)

p = [.3,.3,.4]
μ = 0.
τ = [1.,2.,3.]
pD = .5
pF = .25
pH = 1 - pD - pF
u = .75
m = 3.

o = HnFstuff(df.z, D=length(τ), interpres=300, quadnodes=25)
@time res = optimize(negHnFllSharedμ(o), vcat(SimplextoRⁿ(p),μ,log.(τ),SimplextoRⁿ([pD,pF,pH])...,logit(u),log(m)), LBFGS(), autodiff=:forward)
θ, ll = Optim.minimizer(res), Optim.minimum(res)
# p̂, μ̂ , τ̂ , p̂D, p̂F, p̂H, û, m̂ = RⁿtoSimplex(θ[1:o.D-1]), θ[o.D:2*o.D-1], exp.(θ[2*o.D:3*o.D-1]), RⁿtoSimplex(θ[3*o.D:3*o.D+1])..., logistic(θ[3*o.D+2]), exp(θ[3*o.D+3])
p̂, μ̂ , τ̂ , p̂D, p̂F, p̂H, û, m̂ = RⁿtoSimplex(θ[1:o.D-1]), θ[o.D], exp.(θ[o.D+1:2*o.D]), RⁿtoSimplex(θ[2*o.D+1:2*o.D+2])..., logistic(θ[2*o.D+3]), exp(θ[2*o.D+4])
println((p̂=p̂, μ̂ =μ̂ , τ̂ =τ̂ , p̂D=p̂D, p̂F=p̂F, p̂H=p̂H, û=û, m̂=m̂))
ses = sqrt.(diag(pinv(ForwardDiff.hessian(negHnFllSharedμ(o), θ))))

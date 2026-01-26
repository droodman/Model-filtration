include("Generalized t distribution")

using Distributions, CairoMakie, Random, ThreadsX

# simulate p-hacking round
# z₀ = initial measurement
# σ^2 = true variance of p-hack tries around z₀
# σ₀^2 = prior thereof
# ξ₀ = DOF consumed in forming that prior; i.e., confidence
# B_C = benefit-cost ratio for the research teamn in this study
# m_max = max tries
function sim_phack(;z₀, σ, σ₀, ξ₀, B_C)
  SS = 0.  # sum squares
	k = 0
  σₖ² = σ₀^2
  while (1 - diffcdf(GenT(ξ₀ + k, z₀, √σₖ²), z̄, -z̄)) * B_C > 1
		z₁ₖ = rand(Normal(z₀,σ))
    -z̄ ≤ z₁ₖ ≤ z̄ || return true, k  # success

    SS += (z₁ₖ - z₀)^2
    k += 1
    σₖ² = (ξ₀ * σ₀^2 + SS) / (ξ₀ + k)
	end
  return false, k  # give up
end

Random.seed!(1231)
M = 100_000
zplot = 0:.01:z̄
f = Figure(;size=(750,750), fonts = (; regular = "Computer Modern"))
for (p,σ) ∈ enumerate((.5,1.5))
  Axis(f[2p-1,1]; xlabel=L"z_0", ylabel="Pr[successful p-hacking]", limits=(nothing,nothing,-.05,1.05))
  lines!(zplot, ThreadsX.map(z₀->mean(sim_phack(;z₀, σ, σ₀=2., ξ₀=1., B_C=5)[1] for _ in 1:M), zplot), color=(:blue,.5), colorrange=(.1,.5), linewidth=2, label="Benefit-cost ratio = 16")
  lines!(zplot, ThreadsX.map(z₀->mean(sim_phack(;z₀, σ, σ₀=2., ξ₀=1., B_C=4)[1] for _ in 1:M), zplot), color=(:blue,.4), colorrange=(.1,.5), linewidth=2, label="Benefit-cost ratio = 8")
  lines!(zplot, ThreadsX.map(z₀->mean(sim_phack(;z₀, σ, σ₀=2., ξ₀=1., B_C=3)[1] for _ in 1:M), zplot), color=(:blue,.3), colorrange=(.1,.5), linewidth=2, label="Benefit-cost ratio = 4")
  lines!(zplot, ThreadsX.map(z₀->mean(sim_phack(;z₀, σ, σ₀=2., ξ₀=1., B_C=2)[1] for _ in 1:M), zplot), color=(:blue,.2), colorrange=(.1,.5), linewidth=2, label="Benefit-cost ratio = 2")
  p==1 && axislegend(position=:lt, framevisible = false)
  f[2p-2,1] = Label(f, "σ=$σ", tellwidth=false, fontsize=18)
end
rowgap!(f.layout, 1, 5)
rowgap!(f.layout, 3, 5)
f |> display

save("output/sim p-hacking.png", f)

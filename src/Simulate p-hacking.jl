cd(dirname(@__FILE__))
cd("..")

include("Generalized t distribution.jl")

using Distributions, CairoMakie, Random, ThreadsX, Unzip

# simulate p-hacking round
# z₀ = initial measurement
# σ^2 = true variance of p-hack tries around z₀
# σ₀^2 = prior therefor
# ξ₀ = # of trials taken to form that prior; i.e., confidence
# B_C = benefit-cost ratio for the research team in this study
# M = # of simulations
function sim_phack(;z₀, σ, σ₀, ξ₀, B_C, M)
  successes = 0.
  n = 0
  for _ ∈ 1:M
    SS = 0.  # sum squares
    σₖ² = σ₀^2
    k = 0
    while (1 - diffcdf(GenT(ξ₀ + k, z₀, √σₖ²), z̄, -z̄)) * B_C > 1
      z₁ₖ = rand(Normal(z₀,σ))
      if !(-z̄ ≤ z₁ₖ ≤ z̄)  # success
        successes += 1
        n += k
        break
      end

      SS += (z₁ₖ - z₀)^2
      k += 1
      σₖ² = (ξ₀ * σ₀^2 + SS) / (ξ₀ + k)
    end
    n += k  # give up
  end
  return successes/M, n/M  # success rate and average # of tries
end


Random.seed!(1231)
M = 1_000_000
zplot = 0:.01:z̄
f = Figure(;size=(750,750), fonts = (; regular = "Computer Modern"))
ax = Matrix{Axis}(undef,2,2)
for (p,σ) ∈ enumerate((.5,1.5))
  ax[1,p] = Axis(f[2,p];                                ylabel= p==1 ? "Pr[successful p-hacking]" : "", ylabelsize=18, limits=(nothing,nothing,-.05,1.05))
  ax[2,p] = Axis(f[3,p]; xlabel= L"z_0", xlabelsize=18, ylabel= p==1 ? "Average p-hack tries"     : "", ylabelsize=18, limits=(nothing,nothing,-.5,nothing))
  for (B_C,c) ∈ Iterators.reverse(zip((1,2,3,4,5),(.1,.2,.3,.4,.5)))
    Pr, mean_tries = ThreadsX.map(z₀->sim_phack(;z₀, σ, σ₀=2., ξ₀=1., B_C, M), zplot) |> unzip
    lines!(f[2,p], zplot, Pr        , color=(:blue,c), colorrange=(.1,.5), linewidth=2, label="Benefit-cost ratio = $B_C")
    lines!(f[3,p], zplot, mean_tries, color=(:blue,c), colorrange=(.1,.5), linewidth=2, label="Benefit-cost ratio = $B_C")
  end
  f[1,p] = Label(f, "σ=$σ", tellwidth=false, fontsize=18)
end
axislegend(ax[1,1], position=:lt, framevisible = false)
linkxaxes!(ax[1,:]...)
linkxaxes!(ax[2,:]...)
linkyaxes!(ax[1,:]...)
linkyaxes!(ax[2,:]...)
rowgap!(f.layout, 1, 5.)
f |> display
save("output/sim p-hacking.png", f)

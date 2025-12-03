This is the repo for an ongoing project to model how file-drawering/publication bias and _p_-hacking affect the distribution of published _z_ statistics. The model integrates ideas from [van Zwet, Schwab, and Senn (2021)](https://doi.org/10.1002/sim.9173) and [van Zwet and Gelman (2022)](https://doi.org/10.1080/00031305.2021.1938225) on priors and [Andrews and Kasy (2019)](https://doi.org/10.1257/aer.20180310) on publication bias. It adds _p_-hacking to the mix.

The model is fit with (penalized) maximum likelihood to large data sets of _z_ statistics from ~8 different studies. It is implemented in Julia. Running it should only require [installing Julia](https://julialang.org/downloads), launching Julia, downloading and unzipping this archive, and `include("path/to/archive/src/Model publication distortion.jl")`.

The program produces output like this ("true z" = signal-to-noise ratio):
<img width="799" alt="image" src="https://github.com/droodman/Model-filtration/blob/main/output/GW2%20all.png" />

# Model
<img width="792" height="714" alt="image" src="https://github.com/user-attachments/assets/fd235128-7982-4fa8-8926-9249f1cdf6a9" />

# Likelihood
<img width="1125" height="333" alt="image" src="https://github.com/user-attachments/assets/607b827b-a033-4c74-8aec-f5702396a67b" />




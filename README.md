This is the repo for an ongoing project to model how file-drawering/publication bias and _p_-hacking affect the distribution of published _z_ statistics. The model integrates ideas from [van Zwet, Schwab, and Senn (2021)](https://doi.org/10.1002/sim.9173) and [van Zwet and Gelman (2022)](https://doi.org/10.1080/00031305.2021.1938225) on priors and [Andrews and Kasy (2019)](https://doi.org/10.1257/aer.20180310) on publication bias. It adds _p_-hacking to the mix. [Brodeur et al. (2016)](https://doi.org/10.1257/app.20150044) also estimates both file-drawering and _p_-hacking, in a less parametric way.)

The model is fit with (penalized) maximum likelihood to large data sets of _z_ statistics from ~8 different studies. It is implemented in Julia. Running it should only require downloading and unzipping this archive, [installing](https://julialang.org/downloads) and launching Julia, and `include("path/to/archive/src/Model publication distortion.jl")` in Julia.

The program produces output like this ("true z" = signal-to-noise ratio = (unobserved, true effect)/(observed, true standard error)):
<img width="799" alt="image" src="https://github.com/droodman/Model-filtration/blob/main/output/GW3%20all.png" />

# Model
<img width="775" height="773" alt="image" src="https://github.com/user-attachments/assets/30e24129-d3b1-42ad-8189-8e5da08a0f40" />

For a 50-year-old antecedent, see [Greenwald (1975)](https://faculty.washington.edu/agg/pdf/Gwald_PsychBull_1975.OCR.pdf#page=7), p. 7.

# Likelihood
<img width="748" height="523" alt="image" src="https://github.com/user-attachments/assets/91348728-775b-4fbf-8a15-10b933b444bc" />

# Numerical results
Each column is for a different data set of _z_ statistics:
[Georgescu and Wren (2018)](https://github.com/agbarnett/intervals/blob/master/data/Georgescu.Wren.RData),
[Schuemie et al. (2013)](https://onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002%2Fsim.5925&file=Appendix+G+Revision.xlsx),
[Gerber and Malhotra (2008a)](https://www.nowpublishers.com/article/details/supplementary-info/100.00008024_supp.rar),
[Gerber and Malhotra (2008b)](https://doi.org/10.1177/0049124108318973),
[Brodeur et al. (2016)](https://openicpsr.org/openicpsr/project/113633/version/V1/view?path=/openicpsr/113633/fcr:versions/V1/brodeur_le_sangnier_zylberberg_replication/Data/Final/final_stars_supp.dta&type=file),
[Brodeur, Cook, and Heyes (2020)](https://openicpsr.org/openicpsr/project/120246/version/V1/view?path=/openicpsr/120246/fcr:versions/V1/MM-Data.dta&type=file),
[Arel-Bundock et al. (forthcoming)](https://doi.org/10.1086/734279),
[van Zwet, Schwab, and Senn (2021)](https://osf.io/xq4b2),
[Vivalt (2020)](https://figshare.com/articles/dataset/Replication_files_for_How_Much_Can_We_Generalize_from_Impact_Evaluations_/12048600/1).
```
                                    BW2       BWAbstr2      Setal2     GMpolisci1     GMsoc1         SW2         BCH3        ABetal1       vZSS2         V1    
---------------------------------------------------------------------------------------------------------------------------------------------------------------
p₁                                   0.841         0.84        0.679            1            1        0.707        0.841            1        0.758            1
                                 (0.00496)    (0.00477)     (0.0195)          (0)          (0)      (0.014)     (0.0124)          (0)     (0.0695)          (0)
p₂                                   0.159         0.16        0.321                                  0.293        0.159                     0.242             
                                 (0.00496)    (0.00477)     (0.0195)                                (0.014)     (0.0124)                  (0.0695)             
μ                                     1.36         1.46         1.62         1.16        0.927        0.801            0        0.583       -0.209        0.651
                                 (0.00263)    (0.00309)     (0.0347)     (0.0575)      (0.088)     (0.0209)          (0)     (0.0191)      (0.012)      (0.119)
τ₁                                    1.69         1.67         1.83          1.5          1.4         2.13         1.56          1.6         0.86         1.94
                                 (0.00948)    (0.00955)     (0.0577)     (0.0753)      (0.121)     (0.0456)     (0.0474)     (0.0313)     (0.0508)      (0.159)
τ₂                                    6.07         6.43         7.35                                   7.73         6.87                      2.78             
                                  (0.0987)      (0.105)      (0.146)                                (0.113)      (0.146)                   (0.387)             
ν₁                                    4.02         3.91          4.7         2.19         1.98         7.63         4.61         2.61         5.63          2.1
                                  (0.0518)     (0.0546)        (0.9)      (0.177)      (0.234)       (2.02)      (0.698)     (0.0853)       (2.43)      (0.289)
ν₂                                    24.1         51.5     6.65e+07                               7.35e+07     6.46e+06                      4.71             
                                    (4.14)       (20.4)   (3.21e+09)                             (2.23e+09)          (0)                   (0.713)             
pF                                       0            0            0            0     1.12e-32     1.73e-32        0.113     1.95e-32            0     3.04e-32
                                       (0)          (0)          (0)          (0)   (5.74e-17)   (1.91e-17)     (0.0453)   (2.03e-17)          (0)   (7.14e-17)
pH                                   0.752        0.768        0.713         0.35        0.367        0.213       0.0842         0.21       0.0565       0.0908
                                (0.000884)   (0.000999)     (0.0114)     (0.0426)     (0.0389)     (0.0132)     (0.0173)     (0.0152)     (0.0181)      (0.245)
pD                                   0.183        0.169        0.172        0.425        0.633         0.43        0.802         0.36     5.21e-14            0
                                 (0.00111)     (0.0012)      (0.013)     (0.0748)     (0.0389)     (0.0492)     (0.0319)     (0.0448)   (1.87e-06)          (0)
pR                                  0.0651       0.0638        0.115        0.225     1.76e-11        0.358     1.18e-07         0.43        0.943        0.909
                                (0.000747)   (0.000818)     (0.0105)     (0.0589)   (3.97e-06)     (0.0454)   (8.49e-05)     (0.0437)     (0.0181)      (0.245)
σ                                    0.887        0.889        0.812        0.385        0.646        0.704        0.497        0.663         1.22        0.541
                                 (0.00321)    (0.00366)     (0.0374)       (0.16)      (0.179)     (0.0867)     (0.0954)     (0.0799)      (0.313)       (1.17)
m                                     7.73         7.44         10.8         58.4         38.5         14.1          802         12.9         4.79            1
                                   (0.114)      (0.126)       (2.24)        (142)       (61.8)       (5.74)   (1.75e+03)       (4.92)       (4.71)    (0.00136)
frac_insig_file_drawered                 0            0            0            0      1.3e-32     1.92e-32        0.116     2.18e-32            0     3.31e-32
                                       (0)          (0)          (0)          (0)   (6.68e-17)   (2.11e-17)     (0.0459)   (2.27e-17)          (0)   (7.78e-17)
frac_insig_pubbed_as_is              0.285        0.271        0.344        0.718        0.633        0.824        0.802        0.843        0.963        0.991
                                (0.000808)   (0.000929)    (0.00998)     (0.0329)     (0.0389)     (0.0095)     (0.0319)     (0.0107)    (0.00997)     (0.0244)
p_hacked_frac_of_pubbed_insig        0.298        0.298        0.227       0.0845        0.154        0.058       0.0416       0.0682       0.0275      0.00484
                                (0.000505)   (0.000567)     (0.0047)     (0.0109)     (0.0206)    (0.00346)    (0.00791)    (0.00462)    (0.00762)     (0.0127)
p_hacked_frac_of_sig                0.0507       0.0484       0.0341        0.071       0.0608       0.0198       0.0104       0.0262     7.85e-16            0
                                (0.000302)    (0.00034)    (0.00283)     (0.0129)     (0.0135)    (0.00289)    (0.00236)    (0.00451)   (2.82e-08)          (0)
p_hacked_frac_of_marg_sig            0.483         0.48         0.43        0.216        0.354         0.17        0.131        0.197        0.116       0.0158
                                (0.000512)   (0.000555)    (0.00535)     (0.0232)     (0.0357)    (0.00886)     (0.0246)     (0.0117)     (0.0282)     (0.0422)
H(z₀|z)                               0.42         0.43        0.353        0.168        0.182       0.0831        0.688       0.0797       -0.011      0.00538
                                (0.000644)    (0.00075)    (0.00643)     (0.0201)     (0.0241)    (0.00496)      (0.183)    (0.00737)     (0.0109)     (0.0136)
equiv_sample_reduction               0.253        0.258        0.217         0.11        0.119        0.056        0.379       0.0538     -0.00766      0.00375
---------------------------------------------------------------------------------------------------------------------------------------------------------------
N                                1,295,848      949,159       10,927        1,792          781       28,937       20,189       16,539       23,551          567
---------------------------------------------------------------------------------------------------------------------------------------------------------------
```

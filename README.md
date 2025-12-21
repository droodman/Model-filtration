This is the repo for an ongoing project to model how file-drawering/publication bias and _p_-hacking affect the distribution of published _z_ statistics. The model integrates ideas from [van Zwet, Schwab, and Senn (2021)](https://doi.org/10.1002/sim.9173) and [van Zwet and Gelman (2022)](https://doi.org/10.1080/00031305.2021.1938225) on priors and [Andrews and Kasy (2019)](https://doi.org/10.1257/aer.20180310) on publication bias. It adds _p_-hacking to the mix. ([Brodeur et al. (2016)](https://doi.org/10.1257/app.20150044) also estimates both file-drawering and _p_-hacking, in a less parametric way.)

The model is fit with (penalized) maximum likelihood to large data sets of _z_ statistics from ~8 different studies. It is implemented in Julia. Running it should only require [installing Julia](https://julialang.org/downloads), launching Julia, downloading and unzipping this archive, and `include("path/to/archive/src/Model publication distortion.jl")`.

The program produces output like this ("true z" = signal-to-noise ratio):
<img width="799" alt="image" src="https://github.com/droodman/Model-filtration/blob/main/output/GW2%20all.png" />

# Model
<img width="792" height="714" alt="image" src="https://github.com/user-attachments/assets/fd235128-7982-4fa8-8926-9249f1cdf6a9" />

# Likelihood
<img width="1125" height="333" alt="image" src="https://github.com/user-attachments/assets/607b827b-a033-4c74-8aec-f5702396a67b" />

# Numerical results
Each column is for a different data set of _z_ statistics:
[Georgescu and Wren (2018)](https://github.com/agbarnett/intervals/blob/master/data/Georgescu.Wren.RData),
[Schuemie et al. (2013)](https://onlinelibrary.wiley.com/action/downloadSupplement?doi=10.1002%2Fsim.5925&file=Appendix+G+Revision.xlsx),
[Gerber and Malhotra (2008)](https://www.nowpublishers.com/article/details/supplementary-info/100.00008024_supp.rar),
[Brodeur et al. (2016)](https://openicpsr.org/openicpsr/project/113633/version/V1/view?path=/openicpsr/113633/fcr:versions/V1/brodeur_le_sangnier_zylberberg_replication/Data/Final/final_stars_supp.dta&type=file),
[Brodeur, Cook, and Heyes (2020)](https://openicpsr.org/openicpsr/project/120246/version/V1/view?path=/openicpsr/120246/fcr:versions/V1/MM-Data.dta&type=file),
[Arel-Bundock et al. (forthcoming)](https://doi.org/10.1086/734279),
[van Zwet, Schwab, and Senn (2021)](https://osf.io/xq4b2),
[Vivalt (2020)](https://figshare.com/articles/dataset/Replication_files_for_How_Much_Can_We_Generalize_from_Impact_Evaluations_/12048600/1).
```
-----------------------------------------------------------------------------------------------------------------------------------------------
                                                GW2        Setal2        GM2          SW2         BCH2       ABetal2       vZZS2         V2    
-----------------------------------------------------------------------------------------------------------------------------------------------
p₁                                               0.504       0.437        0.477        0.393        0.501       0.641        0.767        0.511
                                             (0.00332)    (0.0304)     (0.0522)     (0.0167)     (0.0125)    (0.0239)     (0.0192)     (0.0684)
p₂                                               0.496       0.563        0.523        0.607        0.499       0.359        0.233        0.489
                                             (0.00332)    (0.0304)     (0.0522)     (0.0167)     (0.0125)    (0.0239)     (0.0192)     (0.0684)
μ₁                                                1.13        1.15         1.08        0.631       -0.776       0.565       -0.191        0.571
                                             (0.00513)    (0.0518)     (0.0653)     (0.0186)     (0.0525)    (0.0194)     (0.0117)      (0.113)
τ₁                                                1.17        1.21        0.635         1.49        0.106        1.35        0.845         1.09
                                             (0.00684)      (0.07)      (0.211)     (0.0565)      (0.222)    (0.0682)     (0.0428)      (0.227)
τ₂                                                3.34        3.72          3.1         3.75         3.52        3.39         3.26         4.01
                                             (0.00582)    (0.0562)      (0.137)       (0.04)     (0.0435)    (0.0743)     (0.0694)       (0.28)
pF                                               0.222       0.376     6.38e-32     2.28e-32      2.5e-32           0       0.0445     5.06e-28
                                             (0.00673)    (0.0624)    (9.9e-17)   (2.14e-17)   (5.24e-17)         (0)     (0.0559)   (1.42e-14)
pH                                               0.558       0.394        0.359        0.126        0.205       0.207       0.0558       0.0782
                                             (0.00593)    (0.0545)     (0.0421)     (0.0129)     (0.0139)    (0.0155)     (0.0154)      (0.054)
pD                                               0.191       0.178        0.476         0.25        0.795       0.387          0.9     2.25e-15
                                              (0.0014)      (0.02)     (0.0894)     (0.0659)     (0.0139)    (0.0573)     (0.0446)   (1.37e-07)
pR                                              0.0295      0.0518        0.165        0.625     3.24e-13       0.406     1.11e-12        0.922
                                             (0.00159)    (0.0222)     (0.0731)      (0.071)   (9.72e-08)    (0.0561)   (3.06e-13)      (0.054)
σ                                                0.814       0.689        0.683        0.439        0.621       0.749        0.798        0.642
                                             (0.00434)    (0.0556)      (0.162)      (0.155)     (0.0772)     (0.079)      (0.188)      (0.304)
m                                                 9.69        16.8         8.88         40.3          134        11.3         80.4         40.7
                                               (0.188)      (6.12)       (5.76)       (70.8)        (133)      (3.95)        (114)        (133)
fraction_insignificant_file_drawered             0.301        0.46     8.04e-32     2.46e-32     2.62e-32           0       0.0446      5.2e-28
                                             (0.00799)    (0.0616)   (1.25e-16)   (2.31e-17)   (5.51e-17)         (0)     (0.0561)   (1.46e-14)
fraction_insignificant_published_as_is           0.231       0.242        0.684        0.924        0.795       0.836          0.9        0.947
                                              (0.0016)    (0.0161)     (0.0363)     (0.0108)     (0.0139)    (0.0129)     (0.0446)     (0.0417)
fraction_published_insignificant_p_hacked        0.261        0.17        0.113       0.0287        0.106      0.0763        0.041       0.0303
                                             (0.00181)    (0.0167)     (0.0174)     (0.0033)    (0.00684)    (0.0067)    (0.00928)     (0.0249)
fraction_significant_p_hacked                   0.0444       0.026       0.0733      0.00992       0.0254      0.0253      0.00294     3.59e-17
                                            (0.000357)   (0.00315)     (0.0151)    (0.00332)    (0.00412)   (0.00461)    (0.00195)   (2.18e-09)
-----------------------------------------------------------------------------------------------------------------------------------------------
N                                            1,262,710      10,209        1,752       27,465       19,638      16,277       23,474          548
-----------------------------------------------------------------------------------------------------------------------------------------------
```

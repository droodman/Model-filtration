This is the repo for an ongoing project to model how file-drawering/publication bias and _p_-hacking affect the distribution of published _z_ statistics. The model integrates ideas from [van Zwet, Schwab, and Senn (2021)](https://doi.org/10.1002/sim.9173) and [van Zwet and Gelman (2022)](https://doi.org/10.1080/00031305.2021.1938225) on priors and [Andrews and Kasy (2019)](https://doi.org/10.1257/aer.20180310) on publication bias. It adds _p_-hacking to the mix. ([Brodeur et al. (2016)](https://doi.org/10.1257/app.20150044) also estimates both file-drawering and _p_-hacking, in a less parametric way.)

The model is fit with (penalized) maximum likelihood to large data sets of _z_ statistics from ~8 different studies. It is implemented in Julia. Running it should only require downloading and unzipping this archive, [installing](https://julialang.org/downloads) and launching Julia, and `include("path/to/archive/src/Model publication distortion.jl")` in Julia.

The program produces output like this ("true z" = signal-to-noise ratio):
<img width="799" alt="image" src="https://github.com/droodman/Model-filtration/blob/main/output/GW3%20all.png" />

# Model
<img width="775" height="773" alt="image" src="https://github.com/user-attachments/assets/30e24129-d3b1-42ad-8189-8e5da08a0f40" />

# Likelihood
<img width="748" height="523" alt="image" src="https://github.com/user-attachments/assets/91348728-775b-4fbf-8a15-10b933b444bc" />

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
----------------------------------------------------------------------------------------------------------------------------------
                                    GW3        Setal2        GM2         SW2         BCH1       ABetal3       vZZS2         V1    
----------------------------------------------------------------------------------------------------------------------------------
p₁                                   0.496        0.437      0.447        0.707            1       0.641        0.772            1
                                 (0.00332)     (0.0304)    (0.107)      (0.014)          (0)    (0.0239)     (0.0221)          (0)
p₂                                   0.504        0.563      0.553        0.293                    0.359        0.228             
                                 (0.00332)     (0.0304)    (0.107)      (0.014)                 (0.0239)     (0.0221)             
μ₁                                    1.13         1.15       1.09        0.801    -1.12e-14       0.565       -0.192        0.618
                                 (0.00512)     (0.0518)   (0.0663)     (0.0209)      (0.154)    (0.0194)     (0.0118)      (0.119)
τ₁                                    3.34         1.21      0.509         2.13         2.13        1.35        0.817         2.08
                                 (0.00582)       (0.07)    (0.299)     (0.0456)      (0.027)    (0.0682)     (0.0709)      (0.165)
τ₂                                    1.17         3.72       2.94         7.73                     3.39         3.27             
                                 (0.00684)     (0.0562)      (0.4)      (0.113)                 (0.0743)     (0.0779)             
ν₁                                3.07e+08     1.36e+06       7.64         7.63         4.83    5.21e+05         14.6         3.49
                                (5.33e+08)   (3.27e+08)     (14.7)       (2.02)      (0.223)   (1.4e+08)       (30.1)      (0.772)
ν₂                                1.75e+06      6.2e+07       33.9     4.19e+07                 4.27e+07     6.87e+07             
                                (4.81e+07)   (2.13e+09)     (73.5)   (2.11e+09)                (2.1e+09)   (2.92e+09)             
pF                                   0.222        0.376          0            0     2.17e-32           0        0.048      2.7e-10
                                  (0.0067)     (0.0624)        (0)          (0)   (1.71e-17)         (0)     (0.0546)   (4.83e-06)
pH                                   0.558        0.394      0.359        0.213        0.139       0.207       0.0517     2.53e-13
                                 (0.00591)     (0.0545)   (0.0419)     (0.0132)     (0.0141)    (0.0155)     (0.0168)   (1.33e-07)
pD                                   0.189        0.178      0.467         0.43        0.286       0.387          0.9            1
                                 (0.00138)       (0.02)   (0.0885)     (0.0492)      (0.062)    (0.0573)     (0.0432)   (4.83e-06)
pR                                  0.0304       0.0518      0.174        0.358        0.575       0.406     9.65e-06     3.84e-43
                                 (0.00156)     (0.0222)   (0.0737)     (0.0454)     (0.0653)    (0.0561)    (0.00181)   (2.02e-37)
σ                                    0.813        0.689      0.665        0.704        0.284       0.749        0.755            1
                                 (0.00436)     (0.0556)    (0.168)     (0.0867)     (0.0772)     (0.079)      (0.207)          (5)
m                                     9.63         16.8       9.41         14.1          217        11.3          108            1
                                   (0.187)       (6.12)     (6.62)       (5.74)        (490)      (3.95)        (189)       (3.54)
frac_insig_file_drawered              0.43        0.577          0            0      2.5e-32           0       0.0503      2.7e-10
                                 (0.00884)     (0.0548)        (0)          (0)   (1.97e-17)         (0)     (0.0565)   (4.83e-06)
frac_insig_pubbed_as_is              0.248        0.258      0.723        0.869        0.949       0.882          0.9            1
                                 (0.00275)     (0.0249)   (0.0435)     (0.0122)      (0.012)    (0.0155)     (0.0431)   (4.83e-06)
p_hacked_frac_of_pubbed_insig        0.103       0.0504     0.0348       0.0147      0.00411        0.02      0.00495     2.54e-14
                                 (0.00143)    (0.00907)   (0.0111)    (0.00213)    (0.00129)   (0.00311)    (0.00251)   (1.35e-08)
p_hacked_frac_of_sig                 0.126       0.0676      0.129       0.0434       0.0242      0.0528       0.0341     1.09e-13
                                 (0.00132)    (0.00866)   (0.0219)    (0.00462)     (0.0063)   (0.00827)     (0.0115)   (5.74e-08)
p_hacked_frac_of_marg_sig            0.519        0.367      0.333         0.19       0.0918        0.24        0.172     8.22e-14
                                 (0.00428)     (0.0421)   (0.0432)    (0.00984)    (0.00992)    (0.0195)     (0.0508)   (4.32e-08)
----------------------------------------------------------------------------------------------------------------------------------
N                                1,262,710       10,209      1,752       28,937       19,638      16,277       23,474          548
----------------------------------------------------------------------------------------------------------------------------------
```

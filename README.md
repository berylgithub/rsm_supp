# rsm_supp
Datasets and code for "Robust Shepard model for the interpolation of potential energy surfaces"

## Scripts and code
The code is written in [Julia](https://julialang.org/). 
Any `Julia` version can be used to run the code, however, it is highly recommended to use either `Julia 1.7` or `Julia 1.10`.
`main.jl` contains examples on how to call the optimization routine and the evaluation routine.

## Dataset
The pair potential datasets are available in "dataset" folder in `.jld` and `.json` format.
(Only for `Julia`) the dataset in `.jld` format can be loaded by [JLD package](https://github.com/JuliaIO/JLD.jl).
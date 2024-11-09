include("rsm.jl")

using DelimitedFiles
using MLBase, Hyperopt


"""
--------------------------------------------
main functions for optimization and evaluation
--------------------------------------------
"""

"""
The function that does the complete hyperparameter optimization for all datasets.
Inputs:
    - data: a .jld file containing a list of dictionaries, where each dictionary has keys: ["author","mol","V","req","R","state"]
    - (optional) iters: number of iterations for hyperparameter optimization of each dataset, generally larger gives better result but is also longer to finish.
    - (optional) simid: identifier string

Example on calling the function:
    main_hpopt_rsm(load("data/smallmol/hxoy_data_req.jld", "data")[setdiff(1:15, 10)]; iters=270, simid="sim_01", save_folds=true)
"""
function main_hpopt_rsm(data; iters=100, simid="", save_folds=false)
    Random.seed!(603)
    # do fitting for each dataset:
    # for each dataset split kfold
    ld_res = []
    foldss = [] # k-fold storage, in case random gives different numbers (in different machine could happen)
    for j ∈ eachindex(data)
        λ = 0.
        d = data[j]
        req = d["req"]
        println(d["mol"])
        if d["mol"] ∈ ["H4", "H5"] # λ > 0 if H4 or H5 for numerical stability
            println("λ is activated")
            λ = 1e-8
        end
        folds = shuffle.(collect(Kfold(length(d["V"]), 5))) # do 5-folds here
        if save_folds
            fd = Dict("mol" => d["mol"], "folds"=>folds)
            push!(foldss, fd)
        end
        t = @elapsed begin
            ho = @thyperopt for i=iters, # hpspace = 2*3*10*9 = 540
                    sampler = RandomSampler(),
                    #force=[false,true],
                    c=[1,2,3],
                    n_basis = collect(1:10),
                    ptr = LinRange(0.1, 0.9, 9)
                fobj = rosemi_fobj(d["R"], d["V"], req, folds; force=false, c=c, n_basis=n_basis, ptr=ptr, λ = λ)
            end
        end    
        best_params, min_f = ho.minimizer, ho.minimum
        display(best_params)
        display(min_f)
        # save using txt to avoid parallelization crash:
        dj = vcat(d["mol"], d["author"], min_f, t, collect(best_params))
        push!(ld_res, dj)
        out = reduce(vcat,permutedims.(ld_res))
        writedlm("hpopt_rsm_$simid.text", out)
        if save_folds
            save("folds_$simid.jld", "data", foldss)
        end
    end
end


"""
Training and evaluation in for a batch of datasets at once.
Inputs:
    - data: a .jld file containing a list of dictionaries, where each dictionary has keys: ["author","mol","V","req","R","state"]
    - hpopt_params: a .txt file where each row contains a list of hyperparameters for each dataset. Recommended to use the output of `main_hpopt_rsm` function.
    - (optional) sim_id: an identifier string
    - (optional) foldss: a list of dictionaries, each dictionary contains keys: ["folds", "mol"]

Example on calling the function:
    main_eval_rsm(load("data/smallmol/hxoy_data_req.jld", "data")[setdiff(1:15, 10)], readdlm("data/smallmol/hpopt_rsm_hxoy_20241107T124925.text", '\t'); sim_id="sim_01", foldss=load("data/smallmol/folds_hxoy_20241107T124925.jld", "data"))
"""
function main_eval_rsm(data, hpopt_params; sim_id="", foldss=[])
    Random.seed!(603) # still set seed for folds
    # match each result with the corresponding dataset
    λ = 0.
    ld_res = []
    for d ∈ data
        # determine which result id:
        id = findall(d["mol"] .== hpopt_params[:,1])[1]
        hp = hpopt_params[id,:]
        println([d["mol"], hp[5:end]])
        # load data:
        E = d["V"]
        F = rdist.(d["R"], d["req"], c=hp[5]);
        folds = [] # reinitialization
        if isempty(foldss)
            folds = shuffle.(collect(Kfold(length(d["V"]), 5))) # do 5-folds here
        else
            for cfold in foldss
                if cfold["mol"] == d["mol"]
                    folds = cfold["folds"]
                end
            end
        end
        println(hp[5:7])
        MAEs, RMSEs, RMSDs, t_lss, t_preds = rosemi_fitter(F, E, folds; n_basis=hp[6], ptr=hp[7], force=false, λ = λ)
        println(mean(RMSEs))
        # result storage:
        d_res = Dict()
        d_res["MAE"] = MAEs; d_res["RMSE"] = RMSEs; d_res["RMSD"] = RMSDs; d_res["t_train"] = t_lss; d_res["t_test"] = t_preds;
        d_res["mol"] = d["mol"];
        push!(ld_res, d_res)
    end
    save("hdrsm_$sim_id.jld", "data", ld_res)
end

"""
Trains and evaluates one dataset given a pre-optimized hyperparameters.
Inputs:
    - data: a dictionary with keys: ["author","mol","V","req","R","state"]
    - hp: a vector of strings containing a list of hyperparameters for a dataset. Recommended to use the output of `main_hpopt_rsm` function.
    - (optional) folds: a list of lists of integers, containing data point indices.
    - (optional) sim_id: an identifier string

Example on calling the function:
    main_single_eval_rsm(load("data/smallmol/hxoy_data_req.jld", "data")[10], readdlm("data/smallmol/hpopt_rsm_hxoy_20241107T124925.text", '\t')[9,:]; sim_id="H2_2_0711")
"""
function main_single_eval_rsm(data, hp; sim_id="", folds=[], λ = 0.)
    E = data["V"]
    F = rdist.(data["R"], data["req"], c=hp[5])
    if isempty(folds)
        folds = shuffle.(collect(Kfold(length(E), 5)))
    end
    MAEs, RMSEs, RMSDs, t_lss, t_preds = rosemi_fitter(F, E, folds; n_basis=hp[6], ptr=hp[7], force=false, λ = λ)
    println(mean(RMSEs))
    d_res = Dict()
    d_res["MAE"] = MAEs; d_res["RMSE"] = RMSEs; d_res["RMSD"] = RMSDs; d_res["t_train"] = t_lss; d_res["t_test"] = t_preds;
    d_res["mol"] = data["mol"];
    display(d_res)
    save("hdrsm_singlet_$sim_id.jld", "data", d_res)
end
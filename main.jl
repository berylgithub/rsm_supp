include("rsm.jl")

using DelimitedFiles
using Random, Combinatorics
using MLBase, Hyperopt

"""
objective function for hyperparam opt 
"""
function rosemi_fobj(R, E, req, folds; force=true, c=1, n_basis=4, ptr=0.5, λ = 0.)
    F = rdist.(R, req; c=c)
    MAEs, RMSEs, RMSDs, t_lss, t_preds  = rosemi_fitter(F, E, folds; n_basis=n_basis, ptr=ptr, force=force, λ = λ)
    #display(RMSEs)
    return mean(RMSEs) # would mean() be a better metric here? or min() is preferrable?
end


function main_hpopt_rsm(data; iters=100, simid="", save_folds=false)
    Random.seed!(603)
    #data = load("data/smallmol/hxoy_data_req.jld", "data") # load hxoy
    # do fitting for each dataset:
    # for each dataset split kfold
    # possibly rerun with other models?
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
                    force=[false,true],
                    c=[1,2,3],
                    n_basis = collect(1:10),
                    ptr = LinRange(0.1, 0.9, 9)
                fobj = rosemi_fobj(d["R"], d["V"], req, folds; force=force, c=c, n_basis=n_basis, ptr=ptr, λ = λ)
            end
        end    
        best_params, min_f = ho.minimizer, ho.minimum
        display(best_params)
        display(min_f)
        # save using txt to avoid parallelization crash:
        dj = vcat(d["mol"], d["author"], min_f, t, collect(best_params))
        push!(ld_res, dj)
        out = reduce(vcat,permutedims.(ld_res))
        writedlm("data/smallmol/hpopt_rsm_$simid.text", out)
        if save_folds
            save("data/smallmol/folds_$simid.jld", "data", foldss)
        end
    end
end


"""
batch run using params obtained from hpopt
e.g.:
 params = readdlm("data/smallmol/hpopt_hxoy_rsm.text", '\t')
 data = load("data/smallmol/hxoy_data_req.jld", "data")
 sim_id = replace(replace(string(now()), "-"=>""), ":"=>"")[1:end-4]
 main_eval_rsm(data, params; sim_id = sim_id)
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
        F = rdist.(d["R"], d["req"], c=hp[6]);
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
        display(folds)
        MAEs, RMSEs, RMSDs, t_lss, t_preds = rosemi_fitter(F, E, folds; n_basis=hp[7], ptr=hp[8], force=hp[5], λ = λ)
        println(mean(RMSEs))
        # result storage:
        d_res = Dict()
        d_res["MAE"] = MAEs; d_res["RMSE"] = RMSEs; d_res["RMSD"] = RMSDs; d_res["t_train"] = t_lss; d_res["t_test"] = t_preds;
        d_res["mol"] = d["mol"];
        push!(ld_res, d_res)
    end
    save("result/hdrsm_$sim_id.jld", "data", ld_res)
end

"""
singlet eval only
"""
function main_single_eval_rsm(data, hp; sim_id="", folds=[], λ = 0.)
    E = data["V"]
    F = rdist.(data["R"], data["req"], c=hp[2])
    if isempty(folds)
        folds = shuffle.(collect(Kfold(length(E), 5)))
    end
    MAEs, RMSEs, RMSDs, t_lss, t_preds = rosemi_fitter(F, E, folds; n_basis=hp[3], ptr=hp[4], force=hp[1], λ = λ)
    println(mean(RMSEs))
    d_res = Dict()
    d_res["MAE"] = MAEs; d_res["RMSE"] = RMSEs; d_res["RMSD"] = RMSDs; d_res["t_train"] = t_lss; d_res["t_test"] = t_preds;
    d_res["mol"] = data["mol"];
    display(d_res)
    save("result/hdrsm_singlet_$sim_id.jld", "data", d_res)
end
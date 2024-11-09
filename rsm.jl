using JLD, SparseArrays, Distributions, Statistics, StatsBase, ForwardDiff, ReverseDiff, LinearOperators, Krylov
using Random, Combinatorics
using ThreadsX

"""
========================================================================================================================================================================================================================
main engines
========================================================================================================================================================================================================================
"""

"""
The Bspline works for matrices

bspline constructor (mimic of matlab ver by prof. Neumaier)
params:
    - z, feature matrix (ndata, nfeature)
"""
function bspline(z)
    m, n = size(z)
    β = sparse(zeros(m, n))
    z = abs.(z)
    ind = (z .< 1)
    z1 = z[ind]
    β[ind] = 1 .+ 0.75*z1.^2 .*(z1 .- 2)
    ind = (.!ind) .&& (z .< 2)
    z1 = z[ind]
    β[ind] = 0.25*(2 .- z1).^3
    return β
end

"""
mostly unused (although faster), verbose version
"""
function bspline2(x)
    m, n = size(x) # fingerprint x data
    β = sparse(zeros(m, n))
    for j ∈ 1:n 
        for i ∈ 1:m
            z = abs(x[i,j])
            if z < 1
                β[i,j] = 1 + .75*x[i,j]^2 * (z - 2)
            elseif 1 ≤ z < 2
                β[i,j] = 0.25 * (2 - z)^3
            end
        end
    end
    return β
end

"""
Bspline but assumes the input is a scalar, for efficient AD purpose
"""
function bspline_scalar(x)
    β = 0.
    z = abs(x)
    if z < 1
        β = 1 + .75*x^2 * (z - 2)
    elseif 1 ≤ z < 2
        β = .25*(2-z)^3
    end
    return β
end

"""
wrapper to extract M+3 or n_basis amount of splines
params:
    - x, matrix, ∈ Float64 (n_features, n_data) 
    - M, number of basfunc, returns M+3 basfunc
outputs:
    - S, array of basfuncs
        if flatten ∈ Float64 (n_feature*(M+3), n_data)
        else ∈ Float64 (n_feature, n_data, M+3)
"""
function extract_bspline(x, M; flatten=false)
    n_feature, n_data = size(x)
    n_basis = M+3
    S = zeros(n_feature, n_data, n_basis)
    for i ∈ 1:M+3
        S[:, :, i] = bspline(M .* x .+ 2 .- i) # should be M+3 features
    end
    if flatten # flatten the basis
        S = permutedims(S, [1,3,2])
        S = reshape(S, n_feature*n_basis, n_data)
    end
    return S
end

"""
extract both ϕ and dϕ
"""
function extract_bspline_df(x, M; flatten=false, sparsemat=false)
    n_feature, n_data = size(x)
    n_basis = M+3
    if flatten # flatten the basis
        rsize = n_feature*n_basis
        S = zeros(rsize, n_data)
        dϕ = zeros(rsize, n_data)
        @simd for i ∈ 1:n_data
            rcount = 1
            @simd for j ∈ 1:n_basis
                @simd for k ∈ 1:n_feature
                    @inbounds S[rcount, i] = bspline_scalar(M*x[k, i] + 2 - j)
                    @inbounds dϕ[rcount, i] = f_dϕ(M*x[k, i] + 2 - j)
                    rcount += 1
                end
            end
        end
        if sparsemat # naive sparse, could still do smart sparse using triplets (I, J, V)
            S = sparse(S)
            dϕ = sparse(dϕ)
        end
    else # basis in last index of the array, possible for sparse matrix!!
        S = zeros(n_feature, n_data, n_basis)
        dϕ = zeros(n_feature, n_data, n_basis)
        @simd for i ∈ 1:n_basis
            @simd for j ∈ 1:n_data
                @simd for k ∈ 1:n_feature
                    @inbounds S[k, j, i] = bspline_scalar(M*x[k, j] + 2 - i) # should be M+3 features
                    @inbounds dϕ[k, j, i] = f_dϕ(M*x[k, j] + 2 - i)
                end
            end
        end
    end
    return S, dϕ
end

"""
tis uses sparse logic with (I, J, V) triplets hence it will be much more efficient.
"""
function extract_bspline_sparse(x, M; flatten=false)
    n_feature, n_data = size(x)
    n_basis = M+3
    if flatten # flatten the basis
        rsize = n_feature*n_basis
        S = zeros(rsize, n_data)
        dϕ = zeros(rsize, n_data)
        @simd for i ∈ 1:n_data
            rcount = 1
            @simd for j ∈ 1:n_basis
                @simd for k ∈ 1:n_feature
                    @inbounds S[rcount, i] = bspline_scalar(M*x[k, i] + 2 - j)
                    @inbounds dϕ[rcount, i] = f_dϕ(M*x[k, i] + 2 - j)
                    rcount += 1
                end
            end
        end
    end
    #......
end


"""
wrapper for scalar w for ϕ'(w) = dϕ(w)/dw
"""
function f_dϕ(x)
    return ForwardDiff.derivative(bspline_scalar, x)
end

"""
ϕ'(w) = dϕ(w)/dw using AD
params:
    - w, vector of features for a selected data, ∈ Float64 (n_feature) 
output:
    - y := ϕ'(w) ∈ Float64 (n_feature)
"""
function f_dϕ_vec(w)
    y = similar(w)
    for i ∈ eachindex(w)
        y[i] = ForwardDiff.derivative(bspline_scalar, w[i])
    end
    return y
end

"""
more "accurate" basis extractor to the formula: 
    ϕ_l(w) := β_τ((Pw)_t), l = (τ, t), where τ is the basis index, and t is the feature index, P is a scaler matrix (for now I with size of w)
"""
function β_τ(P, w)
    
end

"""
query for
ϕ(w[m], w[k])[l] = ϕ(w[m])[l] - ϕ(w[k])[l] - ϕ'(w[k])[l]*(w[m] - w[k]) is the correct one; ϕ'(w)[l] = dϕ(w)[l]/dw,
currently uses the assumption of P = I hence P*w = w, the most correct one is β':= bspline'((P*w)_t), 
params:
    - l here corresponds to the feature index,
        used also to determine t, which is t = l % n_feature 
        *** NOT NEEDED ANYMORE ***if there exists B basis, hence there are M × l indices, i.e., do indexing of l for each b ∈ B, the indexing formula should be: |l|(b-1)+l, where |l| is the feature length
    - ϕ, basis matrix, ∈ Float64(n_s := n_feature*n_basis, n_data), arranged s.t. [f1b1, f2b1, ...., fnbn]
    - dϕ, the derivative of ϕ, idem to ϕ
    - W, feature matrix, ∈ Float64(n_feature, n_data)
    - m, index of selected unsup data
    - k, ... sup data 
optional:
    - force ∈ bool, whether to include the force condition or not, this is useful for equilibrium geometries
"""
function qϕ(ϕ, dϕ, W, m, k, l, n_feature; force=true)
    if force
        t = l % n_feature # find index t given index l and length of feature vector chosen (or n_f = L/n_b)
        if t == 0
            t = n_feature
        end
        return ϕ[l,m] - ϕ[l, k] - dϕ[l, k]*(W[t,m]-W[t,k]) # ϕ_{kl}(w_m) := ϕ_l(w_m) - ϕ_l(w_k) - ϕ_l'(w_k)(w_m - w_k), for k ∈ K, l = 1,...,L 
    else
        return ϕ[l,m] - ϕ[l, k]
    end
end


"""
for (pre-)computing ϕ_{kl}(w_m) := ϕ_l(w_m) - ϕ_l(w_k) - ϕ_l'(w_k)(w_m - w_k), for k ∈ K (or k = 1,...,M), l = 1,...,L, m ∈ Widx, 
compute (and sparsify outside) B := B_{m,kl}, this is NOT a contiguous matrix hence it is indexed by row and column counter
instead of directly m and kl.
params mostly same as qϕ
"""
function comp_B!(B, ϕ, dϕ, W, Midx, Widx, L, n_feature)
    klc = 1                                                     # kl counter
    for k ∈ Midx
        for l ∈ 1:L
            rc = 1                                              # the row entry is not contiguous
            for m ∈ Widx
                B[rc, klc] = qϕ(ϕ, dϕ, W, m, k, l, n_feature) 
                rc += 1
            end
            klc += 1
        end
    end
end

"""
parallel version of B matrix computation, this allocates to memory inside the function, however it will benefit by the speedup if the CPU count is large 
"""
function comp_Bpar(ϕ, dϕ, W, Midx, Widx, L, n_feature; force=true)
    itcol = Iterators.product(1:L, Midx) # concat block column indices
    its = Iterators.product(Widx, collect(itcol)[:]) # concat indices: row vector with vector form of itcol
    return ThreadsX.map(t-> qϕ(ϕ, dϕ, W, t[1], t[2][2], t[2][1], n_feature; force=force), its)
end

"""
kl index computer, which indexes the column of B
params:
    - M, number of sup data
    - L, n_feature*n_basis
"""
function kl_indexer(M, L)
    klidx = Vector{UnitRange}(undef, M) # this is correct, this is the kl indexer!!
    c = 1:M
    for i ∈ c
        n = (i-1)*L + 1 
        klidx[i] = n:n+L-1
    end
    return klidx
end


"""
compute S_K := ∑1/Dₖ
params:
    - D, mahalanobis distance matrix, ∈ Float64 (n_data, n_data)
    - m, index of the selected unsupervised datapoint
    - Midx, list of index of supervised datapoints, ∈ Vector{Int64}
"""
function comp_SK(D, Midx, m)
    sum = 0.
    for i ∈ eachindex(Midx)
        sum += 1/D[m, i] # not the best indexing way...
    end
    return sum
end


function comp_γk(Dk, SK)
    return Dk*SK
end

function comp_αj(Dj, SK)
    return Dj*SK - 1
end

"""
returns a matrix with size m × j
params:
    SKs, precomputed SK vector ∈ Float64(N)
"""
function comp_γ(D, SKs, Midx, Widx)
    M = length(Midx); N = length(Widx)
    γ = zeros(N, M)
    for kc ∈ eachindex(Midx)
        for mc ∈ eachindex(Widx)
            m = Widx[mc]
            γ[mc, kc] = D[m, kc]*SKs[mc]
        end
    end
    return γ
end

"""
assemble A matrix and b vector for the linear system, should use sparse logic (I, J, V triplets) later!!
params:
    - W, data × feature matrix, Float64 (n_feature, n_data)
    ...
"""
function assemble_Ab(W, E, D, ϕ, dϕ, Midx, Widx, n_feature, n_basis)
    # assemble A (try using sparse logic later!!):
    n_m = length(Midx)
    n_w = length(Widx) # different from n_data!! n_data := size(W)[2]
    n_l = n_feature*n_basis
    rows = n_w*n_m
    cols = n_l*n_m
    A = zeros(rows, cols) 
    b = zeros(rows) 
    rcount = 1 #rowcount
    for m ∈ Widx
        SK = comp_SK(D, Midx, m)
        for j ∈ Midx
            ccount = 1 # colcount
            ∑k = 0. # for the 2nd term of b
            αj = SK*D[j,m] - 1
            for k ∈ Midx
                γk = SK*D[k, m]
                den = γk*αj
                ∑k = ∑k + E[k]/den # E_k/(γk × αj)
                for l ∈ 1:n_l # from flattened feature
                    ϕkl = qϕ(ϕ, dϕ, W, m, k, l, n_feature)
                    #display(ϕkl)
                    num = ϕkl*(1-γk*δ(j, k)) # see RoSemi.pdf and RSI.pdf for ϕ and dϕ definition
                    A[rcount, ccount] = num/den
                    ccount += 1 # end of column loop
                end
            end
            b[rcount] = E[j]/αj - ∑k # assign b vector elements
            rcount += 1 # end of row loop
        end
    end
    return A, b
end

"""
assemble A matrix and b vector for the linear system, with sparse logic (I, J, V triplets), dynamic vectors at first, 3x slower than static ones though
params:
    ...
"""
function assemble_Ab_sparse(W, E, D, ϕ, dϕ, Midx, Widx, n_feature, n_basis)
    n_m = length(Midx)
    n_w = length(Widx) # different from n_data!! n_data := size(W)[2]
    n_l = n_feature*n_basis
    rows = n_w*n_m
    cols = n_l*n_m
    b = zeros(rows)
    J = Vector{Int64}(undef, 0); K = Vector{Int64}(undef, 0); V = Vector{Float64}(undef, 0); # empty vectors
    rcount = 1 #rowcount
    for m ∈ Widx
        SK = comp_SK(D, Midx, m)
        for j ∈ Midx
            ccount = 1 # colcount
            ∑k = 0. # for the 2nd term of b
            αj = SK*D[j,m] - 1
            for k ∈ Midx
                γk = SK*D[k, m]
                den = γk*αj
                ∑k = ∑k + E[k]/den # E_k/(γk × αj)
                for l ∈ 1:n_l # from flattened feature
                    ϕkl = qϕ(ϕ, dϕ, W, m, k, l, n_feature)
                    #display(ϕkl)
                    num = ϕkl*(1-γk*δ(j, k)) # see RoSemi.pdf and RSI.pdf for ϕ and dϕ definition
                    val = num/den
                    # assign the vectors:
                    if val != 0. # check if it's nonzero then push everything
                        push!(J, rcount)
                        push!(K, ccount)
                        push!(V, val)
                    else
                        if (rcount == rows) && (ccount == cols) # final entry trick, push zeros regardless
                            push!(J, rcount)
                            push!(K, ccount)
                            push!(V, val)
                        end
                    end
                    ccount += 1 # end of column loop
                end
            end
            b[rcount] = E[j]/αj - ∑k # assign b vector elements
            rcount += 1 # end of row loop
        end
    end
    A = sparse(J, K, V)
    return A, b
end


"""
predict the energy of w_m by computing V_K(w_m), naive or fair(?) version, since all quantities except phi are recomputed
params:
    - W, fingerprint matrix, ∈Float64(n_feature, n_data)
    - ...
    - m, index of W in which we want to predict the energy
    - n_l := n_basis*n_feature, length of the feature block vector,
output:
    - VK, scalar Float64
notes:
    - for m = k, this returns undefined or NaN by definition of V_K(w).
"""
function comp_VK(W, E, D, θ, ϕ, dϕ, Midx, n_l, n_feature, m)
    SK = comp_SK(D, Midx, m) # SK(w_m)
    RK = 0.
    ccount = 1 # the col vector count, should follow k*l, easier to track than trying to compute the indexing pattern.
    for k ∈ Midx
        ∑l = 0. # right term with l index
        for l ∈ 1:n_l # ∑θ_kl*ϕ_kl
            ϕkl = qϕ(ϕ, dϕ, W, m, k, l, n_feature)
            #kl_idx = n_l*(k-1) + l # use the indexing pattern # doesnt work if the index is contiguous
            θkl = θ[ccount] #θ[kl_idx]  # since θ is in block vector of [k,l]
            ∑l = ∑l + θkl*ϕkl
            #println([ccount, θkl, ϕkl, ∑l])
            ccount += 1
        end
        vk = E[k] + ∑l
        RK = RK + vk/D[k, m] # D is symm
        #println([E[k], ∑l, D[k, m], RK])
    end
    return RK/SK
end

"""
compute Δ_jK(w_m). Used for MAD and RMSD. See comp_VK function, since Δ_jK(w_m) := (VK - Vj)/αj
"""
function comp_ΔjK(W, E, D, θ, ϕ, dϕ, Midx, n_l, n_feature, m, j; return_vk = false)
    SK = comp_SK(D, Midx, m) # compute SK
    RK = 0.
    ∑l_j = 0. # for j indexer, only passed once and j ∈ K
    ccount = 1 # the col vector count, should follow k*l, easier to track than trying to compute the indexing pattern.
    for k ∈ Midx
        ∑l = 0. # right term with l index
        for l ∈ 1:n_l # ∑θ_kl*ϕ_kl
            # for k:
            ϕkl = qϕ(ϕ, dϕ, W, m, k, l, n_feature)
            θkl = θ[ccount] # since θ is in block vector of [k,l]
            θϕ = θkl*ϕkl
            ∑l = ∑l + θϕ
            if k == j # for j terms:
                ∑l_j = ∑l_j + θϕ
            end
            #println([ccount, θkl, ϕkl, ∑l, ∑l_j])
            ccount += 1
        end
        vk = E[k] + ∑l
        RK = RK + vk/D[k, m]
        #println([E[k], ∑l, D[k, m], RK])
    end
    #println(SK)
    VK = RK/SK
    αj = D[j, m]*SK - 1.
    Vj = E[j] + ∑l_j
    #println([VK, Vj, αj])
    if return_vk
        return (VK - Vj)/αj, VK
    else
        return (VK - Vj)/αj
    end
end


"""
overloader for ΔjK, use precomputed distance matrix D and SK[m] 
"""
function comp_v_jm(W, E, D, θ, ϕ, dϕ, SK, Midx, n_l, n_feature, m, j)
    RK = 0.
    ∑l_j = 0. # for j indexer, only passed once and j ∈ K
    ccount = 1 # the col vector count, should follow k*l, easier to track than trying to compute the indexing pattern.
    for k ∈ Midx
        ∑l = 0. # right term with l index
        for l ∈ 1:n_l # ∑θ_kl*ϕ_kl
            # for k:
            ϕkl = qϕ(ϕ, dϕ, W, m, k, l, n_feature)
            θkl = θ[ccount] # since θ is in block vector of [k,l]
            θϕ = θkl*ϕkl
            ∑l = ∑l + θϕ
            if k == j # for j terms:
                ∑l_j = ∑l_j + θϕ
            end
            #println([ccount, θkl, ϕkl, ∑l, ∑l_j])
            ccount += 1
        end
        vk = E[k] + ∑l
        RK = RK + vk/D[k, m]
        #println([E[k], ∑l, D[k, m], RK])
    end
    VK = RK/SK
    αj = D[j, m]*SK - 1.
    Vj = E[j] + ∑l_j
    #println([VK, Vj, αj])
    return (VK - Vj)/αj
end


"""
computes the A*x := ∑_{kl} θ_kl ϕ_kl (1 - γ_k δ_jk)/γ_k α_j
Same as v_j function but for VK only
"""
function comp_Ax_j!(temps, θ, B, Midx, cidx, klidx, γ, α, j, jc)
    ∑k, num, den = temps;
    @simd for c ∈ cidx  # vectorized op on N vector length s.t. x = [m1, m2, ... N]
        k = Midx[c]
        num .= (@view B[:,klidx[c]])*θ[klidx[c]] .* (1 .- (@view γ[:,c]).*δ(j,k))
        den .= (@view γ[:,c]) .* (@view α[:, jc])
        ∑k .+= (num ./ den)
        #∑k .+= /( (@view γ[:,k]) .* (@view α[:, j])) # of length N
    end
end

function comp_Ax!(Ax, Axtemp, temps, θ, B, Midx, cidx, klidx, γ, α)
    # loop for all j:
    for jc ∈ cidx
        comp_Ax_j!(temps, θ, B, Midx, cidx, klidx, γ, α, Midx[jc], jc)
        Axtemp[:, jc] .= temps[1]
        fill!.(temps, 0.)
    end
    Ax .= vec(transpose(Axtemp)) # transpose first for j as inner index then m outer
end

"""
computes b_j := (E_j - ∑_k E_k/γ_k α_j) ∀m for each j, 
"""
function comp_b_j!(temps, E, γ, α, Midx, cidx, j, jc)
    b_j, ∑k = temps;
    for c ∈ cidx
        k = Midx[c]
        @. ∑k = ∑k + (E[k] / (@view γ[:, c])) # ∑_k E_k/γ_k(w_m) , E has absolute index (j, k) while the others are relative indices (jc, mc)
    end
    @. b_j = (E[j] - ∑k) / (@view α[:, jc])
end

function comp_b!(b, btemp, temps, E, γ, α, Midx, cidx)
    for jc ∈ cidx
        comp_b_j!(temps, E, γ, α, Midx, cidx, Midx[jc], jc)
        btemp[:, jc] .= temps[1]
        fill!.(temps, 0.)
    end
    b .= vec(transpose(btemp))  
end

"""
computes Aᵀv, where v ∈ Float64(col of A), required for CGLS
params:
    
"""
function comp_Aᵀv!(Aᵀv, v, B, Midx, Widx, γ, α, L)
    rc = 1 # row counter
    for kc ∈ eachindex(Midx)
        k = Midx[kc]; # absolute index k
        for l ∈ 1:L
            cc = 1 # col counter
            ∑ = 0.
            for mc ∈ eachindex(Widx)
                for jc ∈ eachindex(Midx)
                    j = Midx[jc]; # absolute index j
                    num = B[mc, rc]*v[cc]*(1 - γ[mc,kc]*δ(j,k))
                    den = γ[mc, kc]*α[mc, jc]
                    ∑ = ∑ + num/den
                    cc += 1
                end
            end
            Aᵀv[rc] = ∑
            rc += 1
        end
    end
end

"""
computes ΔjK := ΔjK for m = 1,...,N (returns a vector with length N), with precomputed vector of matrices B instead of (W, ϕ, dϕ)
params:
    - outs, temporary vectors to avoid memalloc
    - E, energy vector, ∈ Float64(n_data)
    - D, distance matrix, ∈ Float64(n_data, n_data)
    - θ, tuning param vec, ∈ Float64(M*L)
    - B, matrix containing ϕ_{m,kl}, ∈ Float64(N, M*L)
    - SKs, vector containing SK ∀m, ∈ Float64(N)
    - Midx, vector containing index of supervised data, ∈ Int(M)
    - Widx, vector containing index of unsupervised data ∈ Int(N)
    - cidx, indexer of k or j, ∈ UnitRange(1:M)
    - klidx, vector containing indexer of block column, ∈ UnitRange(M, 1:L) 
    - αj, vector which contains α_j ∀m, ∈ Float64(N)
    - j, absolute index of j ∈ Midx, Int
output:
    - ΔjK, vector ∀m, ∈ Float64(N) (element of outs vector)
    - VK, VK(w_m) ∀m
"""
function comp_v_j!(outs, E, D, θ, B, SKs, Midx, Widx, cidx, klidx, αj, j)
    ΔjK, VK, vk, vj, RK, ϕkl, ϕjl = outs;
    @simd for c ∈ cidx # vectorized op on N vector length s.t. x = [m1, m2, ... N]
        k = Midx[c]
        ϕkl .= B[:,klidx[c]]*θ[klidx[c]]
        @. vk = E[k] + ϕkl
        @. RK = RK + (vk/D[Widx, c])
        if j == k # for j term
            ϕjl .= ϕkl
        end
    end
    @. VK = RK / SKs
    @. vj = E[j] + ϕjl
    @. ΔjK = (VK - vj) / αj
end


"""
full ΔjK computer ∀jm, m × j matrix
outputs:
    - vmat, matrix ΔjK(w_m) ∀m,j ∈ Float64(N, M) (preallocated outside!)
    - VK, vector containing VK(w_m) ∀m
"""
function comp_v!(v, vmat, VK, outs, E, D, θ, B, SKs, Midx, Widx, cidx, klidx, α)
    # initial loop for VK (the first one cfant be parallel):
    jc = cidx[1]
    comp_v_j!(outs, E, D, θ, B, SKs, Midx, Widx, cidx, klidx, α[:, jc], Midx[jc])
    vmat[:, jc] .= outs[1]
    VK .= outs[2] # this only needs to be computed once
    fill!.(outs, 0.)
    # rest of the loop for ΔjK:
    for jc ∈ cidx[2:end]
        comp_v_j!(outs, E, D, θ, B, SKs, Midx, Widx, cidx, klidx, α[:, jc], Midx[jc])
        vmat[:, jc] .= outs[1]
        fill!.(outs, 0.)
    end
    v .= vec(transpose(vmat))
end

"""
only for VK(w_m) prediction
"""
function comp_VK!(VK, outs, E, D, θ, B, SKs, Midx, Widx, cidx, klidx)
    vk, RK, ϕkl = outs;
    @simd for c ∈ cidx # vectorized op on N vector length s.t. x = [m1, m2, ... N]
        k = Midx[c]
        ϕkl .= B[:,klidx[c]]*θ[klidx[c]]
        @. vk = E[k] + ϕkl
        @. RK = RK + (vk/D[Widx, c])
    end
    @. VK = RK / SKs
end

"""
computes the ΔjK across all m ∈ T (vectorized across m)
"""
function comp_ΔjK!(outs, VK, E, θ, B, klidx, αj, jc, j)
    ΔjK, vj, ϕjl = outs;
    ϕjl .= B[:,klidx[jc]]*θ[klidx[jc]]
    @. vj = E[j] + ϕjl
    @. ΔjK = (VK - vj) / αj
end

"""
computes all of the ΔjK (residuals) given VK for j ∈ K, m ∈ T, indexed by j first then m
"""
function comp_res!(v, vmat, outs, VK, E, θ, B, klidx, Midx, α)
    @simd for jc ∈ eachindex(Midx)
        j = Midx[jc]
        comp_ΔjK!(outs, VK, E, θ, B, klidx, α[:, jc], jc, j)
        vmat[:, jc] .= outs[1]
        fill!.(outs, 0.)
    end
    v .= vec(transpose(vmat))
end

"""
MAD_k(w_m) := 1/|K| ∑_j∈K |ΔjK(w_m)| 
"""
function MAD(W, E, D, θ, ϕ, dϕ, Midx, n_l, n_feature, m)
    len = length(Midx)
    ∑ = 0.
    for j ∈ Midx
        ∑ = ∑ + abs(comp_ΔjK(W, E, D, θ, ϕ, dϕ, Midx, n_l, n_feature, m, j))
    end
    return ∑/len
end

"""
only from a list of ΔjK
"""
function MAD(ΔjKs)
    len = length(ΔjKs)
    return sum(abs.(ΔjKs))/len
end

"""
specialized distances
"""
function fcenterdist(F, T)
    D = zeros(size(F, 1), length(T))
    for j ∈ axes(D, 2)
        for i ∈ axes(D, 1)
            D[i, j] = norm((@view F[i,:]) - (@view F[T[j], :]), 2)^2
        end
    end
    return D
end




"""
========================================================================================================================================================================================================================
intermediate engines
========================================================================================================================================================================================================================
"""

"""
main fitter function, assemble LS -> fit -> save to file
to avoid clutter in main function, called within fitting iters

outputs:
    - indexes of n-maximum MAD
"""
function fitter(F, E, D, ϕ, dϕ, Midx, Tidx, Uidx, Widx, n_feature, mol_name, bsize, tlimit; get_mad=false, get_rmse=false, force=true)
    N = length(Tidx); nU = length(Uidx); nK = length(Midx); Nqm9 = length(Widx)
    nL = size(ϕ, 1); n_basis = nL/n_feature
    #println("[Nqm9, N, nK, nf, ns, nL] = ", [Nqm9, N, nK, n_feature, n_basis, nL])   

    # !!!! using LinearOperators !!!:
    # precompute stuffs:
    t_ab = @elapsed begin
        # indexers:
        klidx = kl_indexer(nK, nL)
        cidx = 1:nK
        # intermediate value:
        SKs_train = map(m -> comp_SK(D, Midx, m), Uidx) # only for training, disjoint index from pred
        γ = comp_γ(D, SKs_train, Midx, Uidx)
        SKs = map(m -> comp_SK(D, Midx, m), Widx) # for prediction
        α = γ .- 1
        B = comp_Bpar(ϕ, dϕ, F, Midx, Uidx, nL, n_feature; force=force) #B = zeros(nU, nK*nL); comp_B!(B, ϕ, dϕ, F, Midx, Uidx, nL, n_feature);
    end
    #println("precomputation time = ",t_ab)
    row = nU*nK; col = nK*nL #define LinearOperator's size
    t_ls = @elapsed begin
        # generate LinOp in place of A!:
        Axtemp = zeros(nU, nK); tempsA = [zeros(nU) for _ in 1:3]
        op = LinearOperator(Float64, row, col, false, false, (y,u) -> comp_Ax!(y, Axtemp, tempsA, u, B, Midx, cidx, klidx, γ, α), 
                                                            (y,v) -> comp_Aᵀv!(y, v, B, Midx, Uidx, γ, α, nL))
        # show(op)
        # generate b:
        b = zeros(nU*nK); btemp = zeros(nU, nK); tempsb = [zeros(nU) for _ in 1:2]
        comp_b!(b, btemp, tempsb, E, γ, α, Midx, cidx)
        # do LS:
        start = time()
        θ, stat = cgls(op, b, itmax=500, verbose=0, callback=CglsSolver -> time_callback(CglsSolver, start, tlimit)) # with callback 🌸
        #θ, stat = cgls(op, b, itmax=500, verbose=0) # without ccallback
    end

    # get residual:
    obj = norm(op*θ - b)^2
    #println("solver obj = ",obj, ", solver time = ",t_ls)

    # get residuals of training set:
    VK = zeros(nU); outs = [zeros(nU) for _ = 1:3]
    comp_VK!(VK, outs, E, D, θ, B, SKs_train, Midx, Uidx, cidx, klidx)
    v = zeros(nU*nK); vmat = zeros(nU, nK); fill!.(outs, 0.)
    comp_res!(v, vmat, outs, VK, E, θ, B, klidx, Midx, α)
    MADs = vec(sum(abs.(vmat), dims=2)) ./ nK # length nU

    # semi-BATCHMODE PRED for Nqm9:
    blength = Nqm9 ÷ bsize # number of batch iterations
    batches = kl_indexer(blength, bsize)
    bend = batches[end][end]
    bendsize = Nqm9 - (blength*bsize)
    push!(batches, bend+1 : bend + bendsize)
    # compute predictions:
    t_batch = @elapsed begin
        VK_fin = zeros(Nqm9)
        #B = zeros(Float64, bsize, nK*nL)
        VK = zeros(bsize); outs = [zeros(bsize) for _ = 1:3]
        @simd for batch in batches[1:end-1]
            B = comp_Bpar(ϕ, dϕ, F, Midx, Widx[batch], nL, n_feature; force=force) #comp_B!(B, ϕ, dϕ, F, Midx, Widx[batch], nL, n_feature)
            comp_VK!(VK, outs, E, D, θ, B, SKs[batch], Midx, Widx[batch], cidx, klidx)
            VK_fin[batch] .= VK
            # reset:
            #fill!(B, 0.); 
            fill!(VK, 0.); fill!.(outs, 0.); 
        end
        # remainder part:
        #B = zeros(Float64, bendsize, nK*nL)
        VK = zeros(bendsize); outs = [zeros(bendsize) for _ = 1:3]
        B = comp_Bpar(ϕ, dϕ, F, Midx, Widx[batches[end]], nL, n_feature; force=force) #comp_B!(B, ϕ, dϕ, F, Midx, Widx[batches[end]], nL, n_feature)
        comp_VK!(VK, outs, E, D, θ, B, SKs[batches[end]], Midx, Widx[batches[end]], cidx, klidx)
        VK_fin[batches[end]] .= VK
        VK = VK_fin # swap
    end
    #println("batchpred time = ",t_batch)

    # get errors: 
    MAE = sum(abs.(VK .- E[Widx])) / Nqm9
    MAE *= 627.503 # convert from Hartree to kcal/mol
    #println("MAE of all mol w/ unknown E is ", MAE)

    RMSE = sqrt(sum((VK .- E[Widx]).^2)/Nqm9) # in original unit, NOT in kcal/mol

    # get the n-highest MAD:
    #= n = 1 # 🌸
    sidxes = sortperm(MADs)[end-(n-1):end]
    MADmax_idxes = Widx[sidxes] # the indexes relative to Widx (global data index) =#
    
    # get min |K| RMSD (the obj func):
    RMSD = obj #Optim.minimum(res)
    
    #println("largest MAD is = ", MADs[sidxes[end]], ", with index = ",MADmax_idxes)
    #println("|K|*∑RMSD(w) = ", RMSD)

    # save also the nK indices and θ's to file!!:
    #data = Dict("centers"=>Midx, "theta"=>θ)
    #save("result/$mol_name/theta_center_$mol_name"*"_$matsize.jld", "data", data)
    # clear variables:
    SKs_train = SKs = γ = α = B = klidx = cidx = Axtemp = tempsA = op = b = tempsb = θ = stat = VK = outs = v = vmat = MADs = batches = VK_fin = nothing; GC.gc()
    # collect the RMS(D,E), D for train, E for test:
    if get_rmse
        return MAE, RMSE, RMSD, t_ls, t_batch
    else
        return MAE, RMSD, t_ls, t_batch #return MAE, MADmax_idxes, t_ls, t_batch
    end
end


"""
converts distance to r^k/(r^k+r0^k), k=1,2,3, r0 ≈ req
"""
function rdist(r,r0;c=1)
    return r^c/(r^c + r0^c)
end

"""
converts distance to e^{-r/r0}
"""
function edist(r,r0)
    return exp(-r/r0)
end



"""
rosemi wrapper fitter, can choose either with kfold or usequence (if kfold = true then pcs isnt used)   
    - F = feature vector (or matrix)
    - E = target energy vector
    - kfold = use kfold or usequence
    - k = number of folds
    - pcs = percentage of centers
    - ptr = percentage of trainings
"""
function rosemi_fitter(F, E, folds; pcs = 0.8, ptr = 0.5, n_basis=4, λ = 0., force=true)
    ndata = length(E)
    #folds = collect(Kfold(ndata, k)) # for rosemi, these guys are centers, not "training set"
    #println(folds)
    MAEs = []; RMSEs = []; RMSDs = []; t_lss = []; t_preds = []
    for (i,fold) in enumerate(folds)
        # fitter:
        ϕ, dϕ = extract_bspline_df(F', n_basis; flatten=true, sparsemat=true)
        #fold = shuffle(fold); 
        centers = fold; lenctr = length(centers) # the center ids
        trids = fold[1:Int(round(ptr*lenctr))] # train ids (K)
        uids = setdiff(fold, trids) # unsupervised ids (U)
        tsids = setdiff(1:ndata, fold)
        D = fcenterdist(F, centers) .+ λ # temp fix for numericals stability
        bsize = max(1, Int(round(0.25*length(tsids)))) # to avoid bsize=0
        MAE, RMSE, RMSD, t_ls, t_pred = fitter(F', E, D, ϕ, dϕ, trids, centers, uids, tsids, size(F, 2), "test", bsize, 900, get_rmse=true, force=force)    
        MAE = MAE/627.503 ## MAE in energy input's unit
        # store results:
        push!(MAEs, MAE); push!(RMSEs, RMSE); push!(RMSDs, RMSD); push!(t_lss, t_ls); push!(t_preds, t_pred); 
    end
    return MAEs, RMSEs, RMSDs, t_lss, t_preds 
end


"""
rerun of HxOy using ROSEMi
"""
function main_rosemi_hxoy(;force=true, c=1, n_basis=4, ptr=0.5)
    Random.seed!(603)
    # pair HxOy fitting:
    data = load("data/smallmol/hxoy_data_req.jld", "data") # load hxoy
    # do fitting for each dataset:
    # for each dataset split kfold
    # possibly rerun with other models?
    ld_res = []
    for i ∈ eachindex(data)[3:3]
        # extract data:
        d = data[i]
        req = d["req"]
        F = rdist.(d["R"], req, c=c) #edist.(d["R"], req) # convert distance features
        E = d["V"]
        println([d["mol"], d["author"], d["state"]])
        folds = shuffle.(collect(Kfold(length(d["V"]), 5))) # do 5-folds here
        MAEs, RMSEs, RMSDs, t_lss, t_preds  = rosemi_fitter(F, E, folds; n_basis=n_basis, ptr=ptr, force=force)
        display([MAEs, RMSEs, RMSDs, t_lss, t_preds ])
        println("RMSE = ", RMSEs)
        # result storage:
        d_res = Dict()
        d_res["MAE"] = MAEs; d_res["RMSE"] = RMSEs; d_res["RMSD"] = RMSDs; d_res["t_train"] = t_lss; d_res["t_test"] = t_preds;
        d_res["mol"] = d["mol"]; d_res["author"] = d["author"]; d_res["state"] = d["state"]; 
        display(d_res)
        push!(ld_res, d_res)
    end
    #save("result/hdrsm_[$force].jld", "data", ld_res)
end


"""
rerun of Hn, 3 ≤ n ≤ 5 molecules using ROSEMI
"""
function main_rosemi_hn(;force=true, c=1, n_basis=5, ptr=0.6) # the hyperparams are from the optimized H2 Kolos
    Random.seed!(603)
    data = load("data/smallmol/hn_data.jld", "data")
    ld_res = []
    for i ∈ setdiff(eachindex(data), [2,3]) # skip partial H4
        λ = 0.
        d = data[i]; F = rdist.(d["R"], 1.401, c=c); # req of H2  
        E = d["V"]
        if d["mol"] == "H5" # for H5, add "regularizer"
            λ = 1e-8
        end
        println(d["mol"])
        folds = shuffle.(collect(Kfold(length(d["V"]), 5))) # do 5-folds here
        MAEs, RMSEs, RMSDs, t_lss, t_preds  = rosemi_fitter(F, E, folds; n_basis=n_basis, ptr=ptr, λ = λ, force=force) 
        display([MAEs, RMSEs, RMSDs, t_lss, t_preds])
        # result storage:
        d_res = Dict()
        d_res["MAE"] = MAEs; d_res["RMSE"] = RMSEs; d_res["RMSD"] = RMSDs; d_res["t_train"] = t_lss; d_res["t_test"] = t_preds;
        d_res["mol"] = d["mol"];
        push!(ld_res, d_res)
    end
    timestamp = replace(replace(string(now()), "-"=>""), ":"=>"")[1:end-4] # exclude ms string
    save("result/hn_rosemi_rerun_$timestamp.jld", "data", ld_res) #save("result/h5_rosemi_rerun_unstable.jld", "data", ld_res) 
end



"""
objective function for hyperparam opt 
"""
function rosemi_fobj(R, E, req, folds; force=true, c=1, n_basis=4, ptr=0.5, λ = 0.)
    F = rdist.(R, req; c=c)
    MAEs, RMSEs, RMSDs, t_lss, t_preds  = rosemi_fitter(F, E, folds; n_basis=n_basis, ptr=ptr, force=force, λ = λ)
    #display(RMSEs)
    return mean(RMSEs) # would mean() be a better metric here? or min() is preferrable?
end
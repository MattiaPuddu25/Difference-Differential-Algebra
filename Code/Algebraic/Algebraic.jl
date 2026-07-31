###############################################################################
#####  Internal Helpers
###############################################################################
function __integer_vector(v::AbstractVector{<:Integer})
    return collect(Int64, v)
end

function __integer_vector_list(
    M::AbstractVector{<:AbstractVector{<:Integer}};
    pad::Bool=false,
)
    V = [__integer_vector(m) for m in M]
    if pad && !isempty(V)
        __pad_integer_vectors!(V)
    end
    return V
end

function __pad_integer_vectors!(
    V::AbstractVector{Vector{Int64}},
    n::Integer=isempty(V) ? 0 : maximum(length, V),
)
    n = Int(n)
    for v in V
        if length(v) < n
            append!(v, zeros(Int64, n - length(v)))
        end
    end
    return V
end

function __pad_variable_masks!(
    L::AbstractVector{<:AbstractVector},
    n::Integer,
)
    n = Int(n)
    out = Vector{BitVector}(undef, length(L))
    for (i, l) in enumerate(L)
        b = BitVector(!iszero(x) for x in l)
        if length(b) < n
            append!(b, falses(n - length(b)))
        end
        out[i] = b
    end
    return out
end

function __multiplicative_variable_masks(
    M::AbstractVector{<:AbstractVector{<:Integer}}
)
    M = __integer_vector_list(M; pad=true)
    isempty(M) && return BitVector[]
    ind = length(M[1])
    n = length(M)
    # Order lexicographically the vectors in M
    perm = sortperm(M)
    sorted_M = M[perm]
    # Initialization
    sorted_MV = [trues(ind) for _ in 1:n]
    for j in 1:ind
        i = 1
        while i <= n
            # Look for the last monomial with the same prefix
            k = i
            while k <= n && @views(sorted_M[k][1:j-1]) == @views(sorted_M[i][1:j-1])
                k += 1
            end
            k -= 1
            # Look for the maximum of j in this block
            max_val = sorted_M[i][j]
            for x in (i+1):k
                if sorted_M[x][j] > max_val
                    max_val = sorted_M[x][j]
                end
            end
            for x in i:k
                if sorted_M[x][j] < max_val
                    sorted_MV[x][j] = false
                end
            end
            i = k + 1
        end
    end
    # Reconstruct the original order in output
    MV = Vector{BitVector}(undef, n)
    for (new_idx, orig_idx) in enumerate(perm)
        MV[orig_idx] = sorted_MV[new_idx]
    end
    return MV
end

@doc raw"""
    __first_janet_missing(
        M::AbstractVector{<:AbstractVector{<:Integer}},
        MV::AbstractVector{<:AbstractVector}
    )
    __first_janet_missing(
        M::AbstractVector{<:AbstractVector{<:Integer}}
    )

Return the first exponent vector which is missing from the Janet cone
    decomposition of `M`, or `nothing` if `M` is Janet complete.

The first method uses the already computed multiplicative-variable masks `MV`.
The second method computes them from `M`.
"""
function __first_janet_missing(
    M::AbstractVector{<:AbstractVector{<:Integer}},
    MV::AbstractVector{<:AbstractVector}
)
    M = __integer_vector_list(M)
    if length(M) != length(MV)
        error("The monomial list and multiplicative-variable list must have the same length")
    end
    isempty(M) && return nothing

    width = maximum([length.(M); length.(MV)])
    __pad_integer_vectors!(M, width)
    MV = __pad_variable_masks!(MV, width)
    for (m, l) in zip(M, MV)
        if all(!iszero, l)
            continue
        end

        for n in findall(iszero, l)
            w = copy(m)
            w[n] += 1
            !cones_check(w, M, MV) && return w
        end
    end
    return nothing
end

function __first_janet_missing(
    M::AbstractVector{<:AbstractVector{<:Integer}}
)
    M = __integer_vector_list(M)
    MV = __multiplicative_variable_masks(M)
    return __first_janet_missing(M, MV)
end

function __variable_masks_from_vars(
    R,
    MV::AbstractVector{<:AbstractVector}
)
    n = nvars(R)
    MV_conv = Vector{BitVector}(undef, length(MV))
    idx_cache = Dict{elem_type(R),Int}()
    for (i, vars) in enumerate(MV)
        r = falses(n)
        for v in vars
            if parent(v) != R
                error("The variables in MV must belong to the same polynomial ring as the monomials")
            end
            idx = get!(idx_cache, v) do
                var_index(v)
            end
            r[idx] = true
        end
        MV_conv[i] = r
    end
    return MV_conv
end

function __check_janet_data(
    w::AbstractVector{<:Integer},
    M::AbstractVector{<:AbstractVector{<:Integer}},
    MV::AbstractVector{<:AbstractVector}
)
    w = __integer_vector(w)
    M = __integer_vector_list(M)
    if length(M) != length(MV)
        error("The monomial list and multiplicative-variable list must have the same length")
    end
    n = length(w)
    if !all(m -> length(m) == n, M)
        error("All monomial exponent vectors must have the same length as w")
    end
    if !all(l -> length(l) == n, MV)
        error("All multiplicative-variable vectors must have the same length as w")
    end
    return nothing
end

function __monomial_from_exponent_vector(
    R,
    e::AbstractVector{<:Integer}
)
    C = MPolyBuildCtx(R)
    push_term!(C, one(coefficient_ring(R)), __integer_vector(e))
    return finish(C)
end

###############################################################################
#####  Functions
###############################################################################

@doc raw"""
    poly_leader(
        p::MPolyRingElem{T}
    ) where T

Return the leader of a polynomial `p` in a polynomial ring endowed with an ordering on the set of variables
    (the greatest variable appearing in `p` with respect to this ordering).\
By convention, constant polynomials have leader 1.

# Examples

```jldoctest
julia> R, (x2, x1) = polynomial_ring(QQ, [:x2, :x1])
julia> p = (x2^2+1)*x1^2+x2
julia> poly_leader(p)
x2


julia> R, (x1, x2) = polynomial_ring(QQ, [:x1, :x2])
julia> p = (x2^2+1)*x1^2+x2
julia> poly_leader(p)
x1
julia> p = R(1)
julia> poly_leader(p)
1
```
"""
function poly_leader(
    p::MPolyRingElem{T}
) where T
    is_constant(p) && return parent(p)(1)
    return maximum(vars(p))
end

######################################################################################################################################
######################################################################################################################################

@doc raw"""
    poly_initial(
        p::MPolyRingElem{T}
    ) where T

Return the initial of a polynomial `p` in a polynomial ring `R` endowed with an ordering on the set of variables
(the coefficient of the leading term of `p` as a univariate polynomial in its leader).

# Examples

```jldoctest
julia> R, (x2, x1) = polynomial_ring(QQ, [:x2, :x1])
julia> p = (x2^2+1)*x1^2+x2
julia> poly_initial(p)
x1^2


julia> R, (x1, x2) = polynomial_ring(QQ, [:x1, :x2])
julia> p = (x2^2+1)*x1^2+x2
julia> poly_initial(p)
x2^2 + 1
```
"""
function poly_initial(
    p::MPolyRingElem{T}
) where T
    if is_constant(p)   # TO BE MODIFIED
        error("The initial is not defined for a constant polynomial")
    end
    ld = poly_leader(p)
    return leading_coefficient(p, var_index(ld))
end

######################################################################################################################################
######################################################################################################################################

@doc raw"""
    multiplicative_variables(
        M::AbstractVector{<:AbstractVector{<:Integer}}
    )
    multiplicative_variables(
        M::AbstractVector{<:MPolyRingElem{T}}
    ) where T

Work in a polynomial ring with variables x\_1>...>x\_l.\
Identify a monomial m = x\_1^(a\_1)\*...\*x\_l^(a\_l) with [a\_1, ..., a\_l].

For a vector `M` = [m\_1, ..., m\_n] of monomials, compute the multiplicative
    variables of each m\_i in `M`.

If `M` is given by exponent vectors, return a (0, 1)-vector `MV`: `MV`[i] has 1
    in position j if x\_j is multiplicative for m\_i.  If `M` is given by monomials
    in a polynomial ring, return the corresponding variables themselves.

If the vectors in `M` do not have the same length, the shorter ones are padded with zeros to reach the maximum length.

# Examples

```jldoctest
julia> M = [[4, 0, 0, 0], [0, 1, 2, 0], [0, 2, 0, 4], [0, 1, 1, 2]]
julia> multiplicative_variables(M)
4-element Vector{Vector{Int64}}:
 [1, 1, 1, 1]
 [0, 0, 1, 1]
 [0, 1, 1, 1]
 [0, 0, 0, 1]


julia> M = [[4], [0, 1, 2], [0, 2, 0, 4], [0, 1, 1, 2]]
julia> multiplicative_variables(M)
4-element Vector{Vector{Int64}}:
 [1, 1, 1, 1]
 [0, 0, 1, 1]
 [0, 1, 1, 1]
 [0, 0, 0, 1]

julia> R, (σ1, σ2, σ3, σ4) = polynomial_ring(QQ, [:σ1, :σ2, :σ3, :σ4])
julia> M = [σ1^4, σ2*σ3^2, σ2^2*σ4^4, σ2*σ3*σ4^2]
julia> multiplicative_variables(M)
4-element Vector{Vector{QQMPolyRingElem}}:
 [σ1, σ2, σ3, σ4]
 [σ3, σ4]
 [σ2, σ3, σ4]
 [σ4]


julia> R, (σ1, σ2) = polynomial_ring(QQ, [:σ1, :σ2])
julia> M=[R(1), σ1, σ2^2, σ1*σ2]
julia> multiplicative_variables(M)
4-element Vector{Vector{QQMPolyRingElem}}:
 []
 [σ1]
 [σ2]
 [σ1, σ2]
```
"""
function multiplicative_variables(
    M::AbstractVector{<:AbstractVector{<:Integer}}
)
    return [Int64.(l) for l in __multiplicative_variable_masks(M)]
end

######################################################################################################################################
######################################################################################################################################

function multiplicative_variables(
    M::AbstractVector{<:MPolyRingElem{T}}
) where T
    isempty(M) && return Vector{Vector{eltype(M)}}()
    if any(p -> length(p) != 1, M)
        error("The elements of the input vector must be monomials")
    end
    R = parent(M[1])
    if any(p -> parent(p) != R, M)
        error("The elements of the input vector must belong to the same polynomial ring")
    end
    exps = [exponent_vector(p, 1) for p in M]
    MV = __multiplicative_variable_masks(exps)
    W = Vector{Vector{elem_type(R)}}(undef, length(exps))
    R_gens = gens(R)

    for i in 1:length(exps)
        W[i] = R_gens[MV[i]]
    end
    return W
end

######################################################################################################################################
######################################################################################################################################

@doc raw"""
    cones_check(
        w::AbstractVector{<:Integer},
        M::AbstractVector{<:AbstractVector{<:Integer}}
    )
    cones_check(
        w::AbstractVector{<:Integer},
        M::AbstractVector{<:AbstractVector{<:Integer}},
        MV::AbstractVector{<:AbstractVector}
    )
    cones_check(
        w::MPolyRingElem{T},
        M::AbstractVector{<:MPolyRingElem{T}}
    ) where T
    cones_check(
        w::MPolyRingElem{T},
        M::AbstractVector{<:MPolyRingElem{T}},
        MV::AbstractVector{<:AbstractVector{<:MPolyRingElem{T}}}
    ) where T

Work in a polynomial ring with variables x\_1>...>x\_l.\
Identify a monomial m = x\_1^(a\_1)\*...\*x\_l^(a\_l) with [a\_1, ..., a\_l].

Given a monomial `w` and a vector `M` of monomials, return `true` if `w` is in one of the cones with vertex in one element of `M`,
    spreading in the directions of the corresponding multiplicative variables, and `false` otherwise.

The monomials may be represented either by exponent vectors or by actual monomials in a polynomial ring.
If `MV` is supplied, it is used as the vector of multiplicative variables; otherwise it is computed from `M`.

# Examples

```jldoctest
julia> M = [[4, 0, 0, 0], [0, 1, 2, 0], [0, 2, 0, 4], [0, 1, 1, 2]]
julia> w = [1, 1, 2, 0]
julia> cones_check(w, M)
false


julia> M = [[4, 0, 0, 0], [0, 1, 2, 0], [0, 2, 0, 4], [0, 1, 1, 2]]
julia> w = [0, 1, 2, 5]
julia> cones_check(w, M)
true


julia> R, (σ1, σ2, σ3, σ4) = polynomial_ring(QQ, [:σ1, :σ2, :σ3, :σ4])
julia> M = [σ1^4, σ2*σ3^2, σ2^2*σ4^4, σ2*σ3*σ4^2]
julia> w = σ1*σ2*σ3^2
julia> cones_check(w, M)
false


julia> R, (σ1, σ2, σ3, σ4) = polynomial_ring(QQ, [:σ1, :σ2, :σ3, :σ4])
julia> M = [σ1^4, σ2*σ3^2, σ2^2*σ4^4, σ2*σ3*σ4^2]
julia> w = σ2*σ3^2*σ4^5
julia> cones_check(w, M)
true
```
"""
function cones_check(
    w::AbstractVector{<:Integer},
    M::AbstractVector{<:AbstractVector{<:Integer}}
)
    w = __integer_vector(w)
    M = __integer_vector_list(M)
    MV = __multiplicative_variable_masks(M)
    return cones_check(w, M, MV)
end

######################################################################################################################################
######################################################################################################################################

function cones_check(
    w::AbstractVector{<:Integer},
    M::AbstractVector{<:AbstractVector{<:Integer}},
    MV::AbstractVector{<:AbstractVector}
)
    w = __integer_vector(w)
    M = __integer_vector_list(M)
    __check_janet_data(w, M, MV)
    for (m, l) in zip(M, MV)
        # Detect the positions in which there is a shift (positive values) and
        # whether there is a position in which a shift is not possible (negative values)
        ok = true
        for i in eachindex(w)
            c = w[i] - m[i]
            if c < 0 || (c > 0 && iszero(l[i]))
                ok = false
                break
            end
        end
        # w is an admissible shift of m
        if ok
            return true
        else
            # w is not an admissible shift of m, but it may be an admissible shift of another monomial
            continue
        end
    end
    return false
end

######################################################################################################################################
######################################################################################################################################

function cones_check(
    w::MPolyRingElem{T},
    M::AbstractVector{<:MPolyRingElem{T}}
) where T
    MV = multiplicative_variables(M)
    return cones_check(w, M, MV)
end

######################################################################################################################################
######################################################################################################################################

function cones_check(
    w::MPolyRingElem{T},
    M::AbstractVector{<:MPolyRingElem{T}},
    MV::AbstractVector{<:AbstractVector{<:MPolyRingElem{T}}}
) where T
    R = parent(w)
    if any(m -> parent(m) != R, M)
        error("The monomials must belong to the same polynomial ring")
    end
    w = exponent_vector(w, 1)
    M = [exponent_vector(m, 1) for m in M]
    MV_conv = __variable_masks_from_vars(R, MV)
    return cones_check(w, M, MV_conv)
end

######################################################################################################################################
######################################################################################################################################

@doc raw"""
    is_janet_complete(
        M::AbstractVector{<:AbstractVector{<:Integer}},
        MV::AbstractVector{<:AbstractVector};
        info::Bool=false,
        missing::Bool=false
    )
    is_janet_complete(
        M::AbstractVector{<:AbstractVector{<:Integer}};
        info::Bool=false,
        missing::Bool=false
    )
    is_janet_complete(
        M::AbstractVector{<:MPolyRingElem{T}},
        MV::AbstractVector{<:AbstractVector{<:MPolyRingElem{T}}};
        info::Bool=false,
        missing::Bool=false
    ) where T
    is_janet_complete(
        M::AbstractVector{<:MPolyRingElem{T}};
        info::Bool=false,
        missing::Bool=false
    ) where T

Work in a polynomial ring with variables x\_1>...>x\_l.\
Identify a monomial m = x\_1^(a\_1)\*...\*x\_l^(a\_l) with [a\_1, ..., a\_l].

For a vector `M` = [m\_1, ...m\_n] of monomials, return `true` if `M` is Janet-complete,
    and return `false` if it is not.

The monomials may be represented either by exponent vectors or by actual monomials in a polynomial ring.
If `MV` is supplied, it is used as the vector of multiplicative variables; otherwise it is computed from `M`.

If the optional argument `info` is set to `true` (which is not its default value),
    a message with the first problem encountered is reported (an element missing in the cone decomposition).

If the optional argument `missing` is set to `true` (which is not its default value),
    the first missing element encountered is also returned.

# Examples

```jldoctest
julia> M = [[4, 0, 0, 0], [0, 1, 2, 0], [0, 2, 0, 4], [0, 1, 1, 2]]
julia> MV = multiplicative_variables(M)
4-element Vector{Vector{Int64}}:
    [1, 1, 1, 1]
    [0, 0, 1, 1]
    [0, 1, 1, 1]
    [0, 0, 0, 1]
julia> is_janet_complete(M,MV)
false
julia> is_janet_complete(M,MV;info=true)
[1, 1, 2, 0] should be added
false
julia> is_janet_complete(M,MV;missing=true)
(false, [1, 1, 2, 0])


julia> M = [[0, 0], [1, 0], [0, 2], [0, 1]]
julia> MV = multiplicative_variables(M)
4-element Vector{Vector{Int64}}:
    [0, 0]
    [1, 1]
    [0, 1]
    [0, 0]
julia> is_janet_complete(M,MV)
true


julia> R, (σ1, σ2, σ3, σ4) = polynomial_ring(QQ, [:σ1, :σ2, :σ3, :σ4])
julia> M = [σ1^4, σ2*σ3^2, σ2^2*σ4^4, σ2*σ3*σ4^2]
julia> is_janet_complete(M; missing=true)
(false, σ1*σ2*σ3^2)


julia> R, (σ1, σ2) = polynomial_ring(QQ, [:σ1, :σ2])
julia> M = [R(1), σ1, σ2^2]
julia> is_janet_complete(M; missing=true)
(false, σ2)
julia> push!(M, σ2);
julia> is_janet_complete(M)
true
```
"""
function is_janet_complete(
    M::AbstractVector{<:AbstractVector{<:Integer}},
    MV::AbstractVector{<:AbstractVector};
    info::Bool=false,
    missing::Bool=false
)
    missing_monomial = __first_janet_missing(M, MV)
    missing_monomial === nothing && return true
    if info == true
        println("$missing_monomial should be added")
    end
    missing == true && return false, missing_monomial
    return false
end

######################################################################################################################################
######################################################################################################################################

function is_janet_complete(
    M::AbstractVector{<:AbstractVector{<:Integer}};
    info::Bool=false,
    missing::Bool=false
)
    M = __integer_vector_list(M)
    MV = __multiplicative_variable_masks(M)
    return is_janet_complete(M, MV; info, missing)
end

######################################################################################################################################
######################################################################################################################################

function is_janet_complete(
    M::AbstractVector{<:MPolyRingElem{T}},
    MV::AbstractVector{<:AbstractVector{<:MPolyRingElem{T}}};
    info::Bool=false,
    missing::Bool=false
) where T
    R = parent(M[1])
    if any(m -> parent(m) != R, M)
        error("The monomials must belong to the same polynomial ring")
    end
    M = [exponent_vector(m, 1) for m in M]
    L = __variable_masks_from_vars(R, MV)

    r = is_janet_complete(M, L; info=false, missing=true)

    r[1] && return true

    if info == true
        p = __monomial_from_exponent_vector(R, r[2])
        println("$p should be added")
    end

    if missing == true
        p = __monomial_from_exponent_vector(R, r[2])
        return false, p
    end

    missing == false && return false
end

######################################################################################################################################
######################################################################################################################################

function is_janet_complete(
    M::AbstractVector{<:MPolyRingElem{T}};
    info::Bool=false,
    missing::Bool=false
) where T
    R = parent(M[1])
    if any(m -> parent(m) != R, M)
        error("The monomials must belong to the same polynomial ring")
    end
    M = [exponent_vector(m, 1) for m in M]
    MV = __multiplicative_variable_masks(M)
    r = is_janet_complete(M, MV; info=false, missing=true)

    r[1] && return true

    if info == true
        p = __monomial_from_exponent_vector(R, r[2])
        println("$p should be added")
    end

    if missing == true
        p = __monomial_from_exponent_vector(R, r[2])
        return false, p
    end

    missing == false && return false
end

######################################################################################################################################
######################################################################################################################################

@doc raw"""
    janet_completion(
        M::AbstractVector{<:AbstractVector{<:Integer}}
    )
    janet_completion(
        M::AbstractVector{<:MPolyRingElem{T}}
    ) where T

Work in a polynomial ring with variables x\_1>...>x\_l.
Identify a monomial m = x\_1^(a\_1)\*...\*x\_l^(a\_l) with [a\_1, ..., a\_l].

For a vector `M` = [m\_1, ...m\_n] of monomials, compute the Janet-completion of `M`.

The monomials may be represented either by exponent vectors or by actual
monomials in a polynomial ring.  The output uses the same representation as the
input.

# Examples

```jldoctest
julia> M = [[4, 0, 0, 0], [0, 1, 2, 0], [0, 2, 0, 4], [0, 1, 1, 2]]
julia> janet_completion(M)
29-element Vector{Vector{Int64}}:
 [4, 0, 0, 0]
 [0, 1, 2, 0]
 [0, 2, 0, 4]
 [0, 1, 1, 2]
 [1, 1, 2, 0]
 [0, 2, 2, 0]
 ⋮
 [3, 2, 2, 0]
 [3, 2, 1, 4]
 [3, 2, 1, 2]
 [2, 2, 1, 3]
 [3, 2, 1, 3]


julia> M = [[0, 0], [1, 0], [0, 2]]
julia> janet_completion(M)
4-element Vector{Vector{Int64}}:
 [0, 0]
 [1, 0]
 [0, 2]
 [0, 1]


julia> R, (σ1, σ2, σ3, σ4) = polynomial_ring(QQ, [:σ1, :σ2, :σ3, :σ4])
julia> M = [σ1^4, σ2*σ3^2, σ2^2*σ4^4, σ2*σ3*σ4^2]
julia> janet_completion(M)
29-element Vector{QQMPolyRingElem}:
 σ1^4
 σ2*σ3^2
 σ2^2*σ4^4
 σ2*σ3*σ4^2
 σ1*σ2*σ3^2
 σ2^2*σ3^2
 ⋮
 σ1^3*σ2^2*σ3^2
 σ1^3*σ2^2*σ3*σ4^4
 σ1^3*σ2^2*σ3*σ4^2
 σ1^2*σ2^2*σ3*σ4^3
 σ1^3*σ2^2*σ3*σ4^3


julia> R, (σ1, σ2) = polynomial_ring(QQ, [:σ1, :σ2])
julia> M = [R(1), σ1, σ2^2]
julia> janet_completion(M)
4-element Vector{QQMPolyRingElem}:
 1
 σ1
 σ2^2
 σ2
```
"""
function janet_completion(
    M::AbstractVector{<:AbstractVector{<:Integer}}
)
    J = __integer_vector_list(M)
    seen = Set(Tuple(m) for m in J)
    while true
        missing_monomial = __first_janet_missing(J)
        if missing_monomial === nothing
            break
        end
        if !(Tuple(missing_monomial) in seen)
            push!(J, missing_monomial)
            push!(seen, Tuple(missing_monomial))
        end
    end
    return J
end

######################################################################################################################################
######################################################################################################################################

function janet_completion(
    M::AbstractVector{<:MPolyRingElem{T}}
) where T
    R = parent(M[1])
    if any(m -> parent(m) != R, M)
        error("The monomials must belong to the same polynomial ring")
    end
    M = [exponent_vector(m, 1) for m in M]
    r = janet_completion(M)
    return [__monomial_from_exponent_vector(R, c) for c in r]
end
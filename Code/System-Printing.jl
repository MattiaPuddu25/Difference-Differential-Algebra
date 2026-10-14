@doc raw"""
    print_splitting_result(result; io=stdout)

Print a splitting result one branch per block, avoiding the full Julia container type display for long vectors of branches.
The input may contain algebraic, difference or differential polynomial vectors.
Equations and inequations are displayed as products of factors when possible.
This changes only their presentation: keep each equation as one constraint and create no branches.\
The optional argument `io` chooses the output stream used for printing.
The default is `stdout`; pass another stream to redirect the formatted display.
"""
function print_splitting_result(
    result::AbstractVector{<:Tuple{<:AbstractVector,<:AbstractVector}};
    io::IO=stdout
)
    if isempty(result)
        println(io, "[]")
        return nothing
    end
    for (i, (E, I)) in enumerate(result)
        println(io, "Branch $i")
        print(io, "  E = ")
        __splitting_print_polynomial_vector(io, E; factorized=true)
        println(io)
        print(io, "  I = ")
        __splitting_print_polynomial_vector(io, I; factorized=true)
        println(io)
    end
    return nothing
end

function __splitting_print_polynomial_vector(
    io::IO,
    polys::AbstractVector;
    factorized::Bool=false
)
    show(IOContext(io, :module => @__MODULE__), eltype(polys))
    print(io, "[")
    for (i, p) in enumerate(polys)
        i > 1 && print(io, ", ")
        print(io, factorized ? __splitting_factor_display(p) : string(p))
    end
    print(io, "]")
    return nothing
end

function __splitting_factor_display(p::Union{MPolyRingElem,ActionPolyRingElem})
    p_clean = p isa MPolyRingElem ? __algebraic_remove_constant_factor(p) : p
    factor_data = Tuple{Bool,String,String}[]
    try
        for term in factor(p_clean)
            f = first(term)
            f = f isa MPolyRingElem ? __algebraic_remove_constant_factor(f) : f
            is_constant(f) && continue
            base = string(f)
            exponent = last(term)
            atomic = __splitting_factor_display_is_atomic(base)
            displayed_base = atomic ? base : "($base)"
            displayed = exponent == 1 ? displayed_base : displayed_base * "^" * string(exponent)
            push!(factor_data, (atomic, base, displayed))
        end
    catch
        return string(p_clean)
    end
    isempty(factor_data) && return string(p_clean)
    if length(factor_data) == 1
        atomic, base, displayed = only(factor_data)
        return displayed == "($base)" ? base : displayed
    end
    sort!(factor_data; by=t -> (t[1] ? 0 : 1, t[2]))
    return join((t[3] for t in factor_data), "*")
end

function __splitting_factor_display_is_atomic(s::AbstractString)
    occursin(" + ", s) || occursin(" - ", s) || return true
    return false
end

###############################################################################
#####  Printing splitting results
###############################################################################

__read_system_vectors(s::AlgebraicSystem) = __algebraic_read_system(s)
__read_system_vectors(s::AbstractActionSystem) = __action_read_system(s)

function print_splitting_result(branches::AbstractVector{<:Union{AlgebraicSystem,AbstractActionSystem}}; io::IO=stdout)
    isempty(branches) && (println(io, "[]"); return nothing)
    __require_same_parent(branches)
    return print_splitting_result([__read_system_vectors(branch) for branch in branches]; io)
end

function Base.show(io::IO, system::AlgebraicSystem)
    print(io, "AlgebraicSystem(", length(getfield(system, :equations)), " equations, ",
          length(getfield(system, :inequations)), " inequations; ",
          )
    if system.simple === true
        print(io, "simple = true")
    else
        print(io, "quasi-simple = ", system.quasi_simple)
    end
    print(io, ")")
end

function Base.show(io::IO, system::DifferenceSystem)
    data = getfield(system, :_data)
    print(io, "DifferenceSystem(", length(data.equations), " equations, ",
          length(data.inequations), " inequations)")
end

function Base.show(io::IO, system::DifferentialSystem)
    data = getfield(system, :_data)
    print(io, "DifferentialSystem(", length(data.equations), " equations, ", length(data.inequations), " inequations; ")
    if system.simple === true
        print(io, "simple = true")
    else
        print(io, "quasi_simple = ", system.quasi_simple)
    end
    print(io, ")")
end
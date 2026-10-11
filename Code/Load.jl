if !isdefined(@__MODULE__, :__DIFFERENCE_DIFFERENTIAL_ALGEBRA_LOADED)

    using Oscar
    include("Polynomial-Systems.jl")
    include("Parent-Checks.jl")

    include("Algebraic/Subresultants.jl")
    include("Algebraic/Algebraic-Simple-Systems.jl")
    include("Algebraic/Algebraic-Thomas-Decomposition.jl")
    include("Algebraic/Combinatorics-Janet.jl")
    include("Algebraic/Covering-Checks.jl")

    include("Action/Action-Janet.jl")
    include("Action/Normal-Form.jl")

    include("Action/Differential/Differential-Simple-Systems.jl")
    include("Action/Differential/Optimizations.jl")
    include("Action/Differential/Differential-Thomas-Decomposition.jl")
    include("System-Printing.jl")

    const __DIFFERENCE_DIFFERENTIAL_ALGEBRA_LOADED = true
end

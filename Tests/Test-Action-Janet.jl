using Oscar, Test, Logging
with_logger(NullLogger()) do
    include(joinpath(@__DIR__, "..", "Code", "Load.jl"))
end

@testset "System Janet results do not expose owned polynomials" begin
    R, u = differential_polynomial_ring(QQ, :u, 1)
    E = [u^2-1]
    system = DifferentialSystem(E)
    key_hash = hash(system)
    masks = multiplicative_variables(system)
    masks[1][1] = false
    @test multiplicative_variables(system) == [trues(1)]
    @test is_janet_complete(system; missing=true)
    for p in (u^2, u[2])
        for reduce in (is_janet_reducible, is_janet_reducible_univariate,
                       (p, s) -> is_janet_reducible(p, s, multiplicative_variables(s)))
            result = reduce(p, system)
            @test result !== false
            @test result.polynomial == only(E)
            zero!(result.polynomial)
            zero!(result.prolongation)
            fill!(result.multiplicative_variables, false)
            @test system.equations == E
            @test hash(system) == key_hash
        end
    end
    @test is_janet_reducible(u, system) === false
end

@testset "BitVector masks throughout Janet computations" begin
    M = [[1, 0], [0, 1]]
    masks = multiplicative_variables(M)
    before = deepcopy(masks)
    @test masks == [BitVector([1, 1]), BitVector([0, 1])]
    for supplied in (masks, view(masks, :))
        @test cones_check([2, 1], M, supplied)
        @test !cones_check([0, 0], M, supplied)
        @test is_janet_complete(M, supplied)
    end
    @test masks == before
    @test_throws MethodError cones_check([2, 1], M, [[1, 1], [0, 1]])
    @test_throws MethodError is_janet_complete(M, [[1, 1], [0, 1]])
    @test multiplicative_variables([Int[], Int[]]) == [BitVector(), BitVector()]
    @test is_janet_complete([[0], [0, 2]], [BitVector([1]), BitVector([1, 1])]; missing=true) == (false, [0, 1])
    @test_throws ErrorException cones_check([1, 0], M, [trues(2)])
    for (constructor, cache) in ((difference_polynomial_ring, __action_janet_data),
                                 (differential_polynomial_ring, __action_janet_data))
        R, (u, v) = constructor(QQ, [:u, :v], 2)
        E = [u[1, 0], u[0, 1], v[0, 0]]
        masks = multiplicative_variables(E)
        @test masks isa Vector{BitVector}
        @test masks == [BitVector([1, 1]), BitVector([0, 1]), BitVector([1, 1])]
        @test multiplicative_variables(elem_type(R)[]) isa Vector{BitVector}
        @test cache(E).multiplicative_variables isa Vector{BitVector}
        # Supplying computed masks reuses the same masks
        @test cache(E, masks).multiplicative_variables === masks
        before = deepcopy(masks)
        for supplied in (masks, view(masks, :))
            @test cache(E, supplied).multiplicative_variables isa Vector{BitVector}
            reduction = is_janet_reducible(u[2, 1], E, supplied)
            @test reduction !== false
            @test reduction.multiplicative_variables isa BitVector
            @test is_janet_reducible(u, E, supplied) === false
            @test is_passive(E, supplied)
        end
        @test is_janet_complete(E)
        @test iszero(normal_form(u[2, 1] + v[1, 1], E).nf)
        @test masks == before
        @test_throws ErrorException cache(E, [trues(2)])
        @test_throws ErrorException cache(E, [trues(1) for _ in E])
        integer_masks = [Int.(mask) for mask in masks]
        @test_throws MethodError cache(E, integer_masks)
        @test_throws MethodError is_janet_reducible(u[2, 1], E, integer_masks)
        @test_throws MethodError is_passive(E, integer_masks)
    end
end

@testset "Fixed-basis differential reductions" begin
    K, t = rational_function_field(QQ, "t")
    coefficient_calls = Ref(0)
    delta = action_derivation(map_from_func(K, K, c -> begin
        coefficient_calls[] += 1
        derivative(c)
    end))
    R, u = differential_polynomial_ring(K, :u, [delta])
    F = [u[1] - t*u]
    data = __action_janet_data(F)

    first_reduction = __is_janet_reducible_univariate(u[2], data)
    @test first_reduction.prolongation == u[2] - u - t*u[1]
    calls_after_first = coefficient_calls[]
    repeated_reduction = __is_janet_reducible_univariate(u[2]^2, data)
    @test repeated_reduction.prolongation == first_reduction.prolongation
    @test coefficient_calls[] == calls_after_first
    @test F == [u[1] - t*u]

    # A changed equation gets fresh Janet data and must not reuse the old derivative.
    refined_data = __action_janet_data([u[1] - 2*t*u])
    refined_reduction = __is_janet_reducible_univariate(u[2], refined_data)
    @test refined_reduction.prolongation == u[2] - 2*u - 2*t*u[1]
    @test coefficient_calls[] > calls_after_first

    # Reduction by h*v on h != 0 amounts to substituting v=0.
    p = (u[1] + u)^3 + (u[1] - u)^2
    reducer = (u + 1)*u[1]
    @test __action_linear_pseudoreduce(p, reducer, u[1]) == u^3 + u^2
    # Preserve the existing pseudo multiplier convention in degree one.
    @test __action_linear_pseudoreduce(u[1] + u, reducer, u[1]) == (u + 1)*u
end

@testset "Janet reduction candidate preserves documented return shape" begin
    R, u = differential_polynomial_ring(QQ, :u, 1)
    du = apply_action(u, 1)
    ddu = apply_action(u, [2])
    reducer = du - u
    data = __action_janet_data([reducer])
    reduction = __is_janet_reducible_univariate(ddu, data)
    @test reduction !== false
    @test propertynames(reduction) == (
        :reducible,
        :alpha,
        :polynomial,
        :prolongation,
        :multiplicative_variables,
    )
    @test !hasproperty(reduction, :index)
    @test !hasproperty(reduction, :variable)
    detailed = __is_janet_reducible_univariate_detailed(ddu, data)
    @test detailed !== false
    @test hasproperty(detailed, :index)
    @test hasproperty(detailed, :variable)
end

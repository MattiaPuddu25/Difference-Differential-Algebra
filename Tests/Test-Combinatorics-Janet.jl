using Oscar, Test, Logging
with_logger(NullLogger()) do
    include(joinpath(@__DIR__, "..", "Code", "Load.jl"))
end

@testset "Janet completion preserves the input monomials" begin
    K, t = rational_function_field(QQ, :t)
    for field in (QQ, GF(7), K)
        R, (x, y, z) = polynomial_ring(field, [:x, :y, :z])
        monomials = [one(R), x, y^2]
        snapshot = deepcopy(monomials)
        @test is_janet_complete(monomials; missing=true) == (false, y)
        @test janet_completion(monomials) == [one(R), x, y^2, y]
        @test monomials == snapshot
    end
end

using Random

@testset "Incremental Janet masks agree with full recomputation" begin
    rng = MersenneTwister(20260915)
    examples = [[[4,0,0,0],[0,1,2,0],[0,2,0,4],[0,1,1,2]],
                [[0],[0,2]], [Int[],Int[]], Vector{Int}[]]
    append!(examples, [[rand(rng,0:3,3) for _ in 1:4] for _ in 1:30])
    for M in examples
        old = deepcopy(M)
        reference = __padded_integer_vectors(M)
        while true
            witness = __first_janet_missing(reference)
            witness === nothing && break
            push!(reference,witness)
        end
        @test janet_completion(M) == reference
        @test M == old
        @test is_janet_complete(reference)
        isempty(reference) && continue
        masks = BitVector[]
        groups = [Dict{Tuple,Tuple{Int,Vector{Int}}}() for _ in eachindex(first(reference))]
        for i in eachindex(reference)
            __janet_insert_mask!(masks,groups,reference[i],i)
            @test masks == __multiplicative_variable_masks(reference[1:i])
        end
    end
end

@testset "Exponent validation and independent padding" begin
    for M in ([[4], [0, 1, 2]], [Int32[4], Int32[0, 1, 2]],
              [big.([4]), big.([0, 1, 2])], [view([4, 8], 1:1), view([0, 1, 2], :)])
        before = deepcopy(M)
        masks = multiplicative_variables(M)
        @test masks isa Vector{BitVector}
        @test masks == [[1, 1, 1], [0, 1, 1]]
        @test M == before
    end
    M = [[0], [0, 2]]
    before = deepcopy(M)
    @test is_janet_complete(janet_completion(M))
    @test M == before
    @test_throws ArgumentError multiplicative_variables([[1], [-1]])
    @test multiplicative_variables(Vector{Int}[]) isa Vector{BitVector}
    @test isempty(multiplicative_variables(Vector{Int}[]))
    @test __integer_check(Int32[1, 2]) == Int32[1, 2]
end

@testset "Literature: Janet division (Robertz 2018, Examples 2.17 and 2.19)" begin
    M = [
        [2, 1, 0],
        [2, 0, 1],
        [0, 2, 1],
        [0, 1, 2]
    ]

    expected_MV = [
        BitVector([true,  true,  true ]),
        BitVector([true,  false, true ]),
        BitVector([false, true,  true ]),
        BitVector([false, false, true ])
    ]

    @test multiplicative_variables(M) == expected_MV
    @test !is_janet_complete(M)

    expected_completion = [
        [2, 1, 0],
        [2, 0, 1],
        [0, 2, 1],
        [0, 1, 2],
        [1, 2, 1],
        [1, 1, 2],
    ]

    J = janet_completion(M)
    @test Set(Tuple.(J)) == Set(Tuple.(expected_completion))
    @test is_janet_complete(J)

    masks = Dict(Tuple(m) => mv for (m, mv) in zip(J, multiplicative_variables(J)))
    @test masks[(2, 1, 0)] == BitVector([true,  true,  true ])
    @test masks[(2, 0, 1)] == BitVector([true,  false, true ])
    @test masks[(1, 2, 1)] == BitVector([false, true,  true ])
    @test masks[(1, 1, 2)] == BitVector([false, false, true ])
    @test masks[(0, 2, 1)] == BitVector([false, true,  true ])
    @test masks[(0, 1, 2)] == BitVector([false, false, true ])
end
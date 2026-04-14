using Test
using Random
using PhaseTypeDistributions
using PhaseTypeDistributions: mgf
using Distributions
using LinearAlgebra
using Statistics

Random.seed!(42)

@testset "PhaseTypeDistributions.jl" begin
    include("test_phdist.jl")
    include("test_subtypes.jl")
    include("test_conversions.jl")
    include("test_comparison.jl")
end

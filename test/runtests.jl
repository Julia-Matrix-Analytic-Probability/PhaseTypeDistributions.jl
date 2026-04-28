using Test
using StableRNGs
using PhaseTypeDistributions
using PhaseTypeDistributions: mgf
using Distributions
using LinearAlgebra
using Statistics

# Per-testset `rng = StableRNG(seed)` is preferred over a global seed: it makes
# each random testset independently reproducible and order-insensitive.

@testset "PhaseTypeDistributions.jl" begin
    include("test_phdist.jl")
    include("test_subtypes.jl")
    include("test_conversions.jl")
    include("test_comparison.jl")
    include("test_maph.jl")
end

module PhaseTypeDistributions

using LinearAlgebra
using Random
using Statistics
using Distributions
import Distributions: pdf, logpdf, cdf, insupport, minimum, maximum
import Random: rand
import Statistics: mean, var
using SpecialFunctions: logabsgamma

# Abstract type and general PHDist
include("PHDist.jl")

# Specialized subtypes
include("HyperExponentialDist.jl")
include("HypoExponentialDist.jl")
include("ErlangPHDist.jl")
include("CoxianDist.jl")

# Conversions between types
include("conversions.jl")

# Comparison helpers for non-identifiable distributions
include("comparison.jl")

# Types
export AbstractPHDist, PHDist
export HyperExponentialDist, HypoExponentialDist, ErlangPHDist, CoxianDist

# Accessor functions
export initial_prob, subgenerator, exit_rates, nphases

# PH-specific functions
export scv, kth_moment, mgf

# Comparison helpers
export moments_isapprox, distribution_isapprox, moment_vector

end

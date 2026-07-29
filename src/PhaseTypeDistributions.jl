module PhaseTypeDistributions

using LinearAlgebra
using Random
using Statistics
using Distributions
using FixedSparsityMatrices
import Distributions: pdf, logpdf, cdf, ccdf, logccdf, insupport, minimum, maximum,
    quantile, params, skewness, kurtosis, mgf
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

# Multi-absorbing phase-type distributions
include("MAPHDist.jl")

# Types
export AbstractPHDist, PHDist
export HyperExponentialDist, HypoExponentialDist, ErlangPHDist, CoxianDist
export AbstractMAPHDist, MAPHDist

# Accessor functions
export initial_prob, subgenerator, exit_rates, nphases
export exit_rate_matrix, nabsorbing, absorption_probs, marginal_absorption

# PH-specific functions (mgf extends Distributions.mgf — not re-exported)
export scv, kth_moment

# MAPH-specific functions
export kth_joint_moment, conditional_time, rand_censored

# Comparison helpers
export moments_isapprox, distribution_isapprox, moment_vector

# Re-export the fixed-sparsity array types (the storage backing α, T, D)
export FixedSparsityMatrix, FixedSparsityVector, pattern

end

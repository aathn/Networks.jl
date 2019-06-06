#=
This file contains activation and cost function options for neural
networks. Intended for use with Networks.jl, which specifies the
following abstract types:
- abstract type ActivationFunc end
- abstract type CostFunc end
=#

# Activation functions
struct Sigmoid <: ActivationFunc end
(σ::Sigmoid)(z) = 1.0/(1.0+exp(-z))
activation_deriv(σ::Sigmoid, z) = σ(z)*(1-σ(z))

struct Linear <: ActivationFunc end
(l::Linear)(z) = z
activation_deriv(l::Linear, z) = 1.

# Cost functions
struct Quadratic <: CostFunc end
(c::Quadratic)(a, y) = 0.5*norm(a.-y)^2
delta(c::Quadratic, act::ActivationFunc, z, a, y) = (a-y).*activation_deriv(act, z)

struct CrossEntropy <: CostFunc end
(c::CrossEntropy)(a, y) = sum(-y .* log.(a) .- (1 .- y) .* log.(1 .- a))
delta(c::CrossEntropy, act::ActivationFunc, z, a, y) = (a-y)

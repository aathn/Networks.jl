module Networks
using Random
using LinearAlgebra
using Statistics
export
    sequential,
    feedforward,
    SGDtrain,
    Sigmoid,
    Linear,
    Quadratic,
    CrossEntropy
    #=,
    loadNet,
    saveNet
    =#
abstract type AbstractNet end
abstract type ActivationFunc end
abstract type CostFunc end
Broadcast.broadcastable(a::ActivationFunc) = Ref(a)
Broadcast.broadcastable(c::CostFunc) = Ref(c)

include("sequential.jl")
include("netfuncs.jl")
# include("netio.jl")
end

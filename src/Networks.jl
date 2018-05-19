module Networks
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

include("sequential.jl")
include("netfuncs.jl")
# include("netio.jl")
end

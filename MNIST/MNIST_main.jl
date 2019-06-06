include("../src/Networks.jl")
include("MNIST_load.jl")
using Main.Networks

function main()
    net = sequential([784, 30, 10], Sigmoid(), CrossEntropy())
    traindata, testdata = loaddata()
    SGDtrain(net, traindata, 30, 10, 0.5, test_data = testdata, λ = 5.0, verbose = true)
end
main()

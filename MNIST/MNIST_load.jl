using MLDatasets
function loaddata(rng = 1:50000)
    train_x, train_y = MNIST.traindata(Float64, Vector(rng))
    train_x = [train_x[:,:,x][:] for x in 1:size(train_x, 3)]
    train_y = [vectorize(x) for x in train_y]
    traindata = [(x, y) for (x, y) in zip(train_x, train_y)]

    test_x, test_y = MNIST.testdata(Float64)
    test_x = [test_x[:,:,x][:] for x in 1:size(test_x, 3)]
    testdata = [(x, y) for (x, y) in zip(test_x, test_y)]
    return traindata, testdata
end

function vectorize(n)
    ev = zeros(10)
    ev[n+1] = 1
    return ev
end

struct sequential{A <: ActivationFunc, C <: CostFunc} <: AbstractNet
    sizes::Array{Int64,1}
    biases::Array{Array{Float64,1},1}
    weights::Array{Array{Float64,2},1}
    activation::A
    cost::C
end

function sequential(sizes, activation, cost)
    biases = [randn(y) for y in sizes[2:end]]
    weights = [randn(y, x) for (x, y) in zip(sizes[1:end-1], sizes[2:end])]

    sequential(sizes, biases, weights, activation, cost)
end

# Used to cache arrays for the backpropagation
struct backprop_cache
    weights_transp::Array{Array{Float64,2},1}
    activations::Array{Array{Float64,2},1}
    a_transp::Array{Array{Float64,2},1}
    zs::Array{Array{Float64,2},1}
    δs::Array{Array{Float64,2},1}
    ∇_b::Array{Array{Float64,2},1}
    ∇_w::Array{Array{Float64,2},1}
    X::Array{Float64,2}
    Y::Array{Float64,2}
end

function backprop_cache(sizes, mini_batch_size)
    weights_transp = [randn(x, y) for (x, y) in zip(sizes[1:end-1], sizes[2:end])]
    activations = [zeros(y, mini_batch_size) for y in sizes]
    a_transp = [zeros(mini_batch_size, y) for y in sizes[1:end-1]]
    zs = [zeros(y, mini_batch_size) for y in sizes[2:end]]
    δs = [zeros(y, mini_batch_size) for y in sizes[2:end]]
    ∇_b = [zeros(y, mini_batch_size) for y in sizes[2:end]]
    ∇_w = [zeros(y, x) for (x, y) in zip(sizes[1:end-1], sizes[2:end])]
    X = zeros(sizes[1], mini_batch_size)
    Y = zeros(sizes[end], mini_batch_size)

    backprop_cache(weights_transp, activations, a_transp, zs, δs, ∇_b, ∇_w, X, Y)
end

function SGDtrain(net::sequential, training_data, epochs, mini_batch_size, η; 
    test_data=nothing, λ=0.0, verbose = false)
    n_test = test_data != nothing ? length(test_data) : nothing
    n = length(training_data)
    bcache = backprop_cache(net.sizes, mini_batch_size)

    for j in 1:epochs
        time_taken = @elapsed begin
            training_data .= shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size-1] for k in 1:mini_batch_size:n]

            for batch in mini_batches
                grad_desc(net, bcache, batch, η, λ, n)
            end
        end 

        if verbose
            if test_data != nothing
                println("Epoch ", j,": ", evaluate(net, test_data), "/", n_test)
            else
                println("Epoch ", j," complete.")
            end
            println("Time elapsed: $time_taken")
        end
    end
end

function feedforward(net::sequential, a)
    for (w, b) in zip(net.weights, net.biases)
        a = net.activation.(w*a .+ b)
    end
    return a
end

function evaluate(net::sequential, test_data)
    return sum(findmax(feedforward(net,x))[2] - 1 == y for (x, y) in test_data)
end

function grad_desc(net::sequential, bcache::backprop_cache, batch, η, λ, n)
    for i in 1:length(batch)
        bcache.X[:,i] .= batch[i][1]
        bcache.Y[:,i] .= batch[i][2]
    end

    backprop!(net, bcache)

    ∇_b = bcache.∇_b
    ∇_w = bcache.∇_w

    for i in 1:length(net.biases)
        net.biases[i] .-= η.*vec(mean(∇_b[i];dims=2))
    end

    l2fac = 1.0-η*λ/n
    avgfac = η/length(batch)
    for i in 1:length(∇_w)
        net.weights[i] .= l2fac.*net.weights[i] .- avgfac.*∇_w[i]
    end
end

function backprop!(net::sequential, bcache::backprop_cache)
    X = bcache.X
    Y = bcache.Y
    ∇_b = bcache.∇_b
    ∇_w = bcache.∇_w
    zs = bcache.zs
    δs = bcache.δs
    activations = bcache.activations
    a_transp = bcache.a_transp
    w_transp = bcache.weights_transp

    activations[1] .= X
    num_layers = length(net.sizes)

    for i in 1:num_layers-1
        b = net.biases[i]; w = net.weights[i]
        mul!(zs[i], w, activations[i])
        zs[i] .+= b
        activations[i+1] .= net.activation.(zs[i])
    end

    δ = δs[end]
    δ .= delta.(net.cost, net.activation, zs[end], activations[end], Y)
    ∇_b[end] .= δ
    transpose!(a_transp[end], activations[end-1])
    mul!(∇_w[end], δ, a_transp[end])

    for l in 1:num_layers-2
        z = zs[end-l]
        δ = δs[end-l]
        transpose!(w_transp[end-l+1], net.weights[end-l+1])
        mul!(δ, w_transp[end-l+1], δs[end-l+1])
        δ .*= activation_deriv.(net.activation, z)

        ∇_b[end-l] .= δ
        transpose!(a_transp[end-l], activations[end-l-1])
        mul!(∇_w[end-l], δ, a_transp[end-l])
    end
    return nothing
end

using FileIO
using Plots
gr()

function plotdigit(A)
    a1 = reshape(A, 28, 28)
    display(heatmap(a1', yflip = true, legend = false, c = :grays))
end

function classifyImg(net::sequential, filename, plt = false)
    raw = load(filename)
    img = 1-Gray.(vec(raw'))
    if plt
        plotdigit(img)
    end
    return findmax(feedforward(net, img))[2] - 1
end

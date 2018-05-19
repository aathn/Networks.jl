using JLD
function loadNet(filename)
    try
        net = jldopen(filename, "r") do file
            read(file, "net")
        end
        return net
    catch
        throw(error("Failed loading net from ", filename))
    end
end

function saveNet(net, filename)
    try
        jldopen(filename, "w") do file
            write(file, "net", net)
        end
    catch
        throw(error("Failed saving net to ", filename))
    end
end

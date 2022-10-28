module Visualization

using TSne
using Plots

export visualization2D, visualization3D

function findindex(token::String, all_tokens::Vector{Any}, sizeX::Tuple)
    indexes = []

    for seq in 1:sizeX[1]
        for (index, value) in enumerate(all_tokens[seq])
            if occursin(token, value)
                append!(indexes,sizeX[2]*(seq-1) + index)
            end
        end
    end

    return indexes
end

function visualization2D(all_tokens, X)
    data = X.numpy()
    
    sizeX = size(data)
    arr = Matrix{Float64}(undef, sizeX[1]*sizeX[2], sizeX[3])

    for i in 1:sizeX[1]
        for j in 1:sizeX[2]
            arr[(i-1)*(sizeX[2])+j,:] = data[i,j,:] 
        end
    end

    tsne_result = tsne(arr, 2, 768, 1000, 20.0);

    token1 = "πατερα"
    token2 = "μητερα"
    indexes1 = findindex(token1, all_tokens, sizeX)
    indexes2 = findindex(token2, all_tokens, sizeX)

    x1 = []
    y1 = []
    for i in indexes1
        append!(x1, tsne_result[i,1])
        append!(y1, tsne_result[i,2])
    end 

    x2 = []
    y2 = []
    for i in indexes2
        append!(x2, tsne_result[i,1])
        append!(y2, tsne_result[i,2])
    end 

    tsne_result = tsne_result[setdiff(1:end, indexes1), :]
    tsne_result = tsne_result[setdiff(1:end, indexes2), :]

    theplot = scatter(tsne_result[:,1], tsne_result[:,2], marker=(2,2,:auto,stroke(0)),color=:yellow)
    theplot = scatter!(x1, y1, marker=(2,2,:auto,stroke(0)), color=:red, label=token1)
    theplot = scatter!(x2, y2, marker=(2,2,:auto,stroke(0)), color=:blue, label=token2)

    Plots.pdf(theplot, "./../txt_files/text_delta/juliaResults/myplot2D.pdf")

end

function visualization3D(all_tokens, X)
    data = X.numpy()
    
    sizeX = size(data)
    arr = Matrix{Float64}(undef, sizeX[1]*sizeX[2], sizeX[3])

    for i in 1:sizeX[1]
        for j in 1:sizeX[2]
            arr[(i-1)*(sizeX[2])+j,:] = data[i,j,:] 
        end
    end

    tsne_result = tsne(arr, 3, 768, 1000, 20.0);

    token1 = "πατερα"
    token2 = "μητερα"
    indexes1 = findindex(token1, all_tokens, sizeX)
    indexes2 = findindex(token2, all_tokens, sizeX)

    x1 = []
    y1 = []
    z1 = []
    for i in indexes1
        append!(x1, tsne_result[i,1])
        append!(y1, tsne_result[i,2])
        append!(z1, tsne_result[i,3])
    end 

    x2 = []
    y2 = []
    z2 = []
    for i in indexes2
        append!(x2, tsne_result[i,1])
        append!(y2, tsne_result[i,2])
        append!(z2, tsne_result[i,3])
    end 

    tsne_result = tsne_result[setdiff(1:end, indexes1), :]
    tsne_result = tsne_result[setdiff(1:end, indexes2), :]

    theplot = scatter(tsne_result[:,1], tsne_result[:,2], tsne_result[:,3], marker=(2,2,:auto,stroke(0)),color=:yellow)
    theplot = scatter!(x1, y1, z1, marker=(2,2,:auto,stroke(0)), color=:red, label=token1)
    theplot = scatter!(x2, y2, z2, marker=(2,2,:auto,stroke(0)), color=:blue, label=token2)

    Plots.pdf(theplot, "./../txt_files/text_delta/juliaResults/myplot3D.pdf")
end

end
include("preprocessing.jl")
include("createEmbeddings.jl")
include("visualization.jl")


import .Preprocessing
import .CreateEmbeddings
import .Visualization


# edit.txt: a sentence in each row
print("Text preprocessing... ")
Preprocessing.textPreprocessing("./../txt_files/text_delta/text_delta.txt", "./txt_files/text_delta/juliaResults/edit.txt")
print("Done.")

(all_tokens, x) = CreateEmbeddings.createEmbeddings("./../txt_files/text_delta/juliaResults/edit.txt");

Visualization.visualization2D(all_tokens, x)

Visualization.visualization3D(all_tokens, x)
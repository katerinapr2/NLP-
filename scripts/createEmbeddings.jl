module CreateEmbeddings

using PyCall
using Tensors
@pyimport torch
transformers = pyimport("transformers")
np = pyimport("numpy")

export createEmbeddings

include("preprocessing.jl")
import .Preprocessing

function createEmbeddings(file::String)

    # if torch.cuda.is_available()
    #     device = torch.device("cuda:0")
    # else
    #     device = torch.device("cpu")
    # end

    tokenizer = transformers.AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
    model = transformers.AutoModel.from_pretrained("nlpaueb/bert-base-greek-uncased-v1", output_hidden_states = true)
    

    (input_ids, attention_masks) = Preprocessing.inputs(file, tokenizer)

    all_tokens = []
    for i in input_ids
        push!(all_tokens, tokenizer.convert_ids_to_tokens(i, skip_special_tokens=false))
    end

    input_tensor = torch.tensor(np.array(input_ids))
    att_tensor = torch.tensor(np.array(attention_masks))
    
    let outputs = []
        @pywith torch.no_grad() begin 
            outputs = model(input_tensor,att_tensor)
        end
    
        pooler_output = outputs["pooler_output"]
        hidden_states = outputs["hidden_states"]
        last_hidden_state = outputs["last_hidden_state"]


        print("Number of layers: $(length(hidden_states)) (initial embeddings + 12 BERT layers)\n")
        layer_i = 1
        print("Number of batches: $(length(hidden_states[layer_i]))\n")
        batch_i = 1
        print("Number of tokens: $(length(hidden_states[layer_i][batch_i]))\n")
        token_i = 1
        print("Number of hidden units: $(length(hidden_states[layer_i][batch_i][token_i]))\n")


        return all_tokens, last_hidden_state
        
    end

end

end
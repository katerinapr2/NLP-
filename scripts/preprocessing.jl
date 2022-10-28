module Preprocessing

using Unicode
using PyCall

export preprocessing, inputs

function  strip_accents_and_lowercase(s::String)
    text = ""
    for c in Unicode.normalize(s, stripmark=true, casefold=true)
        text = text * c
    end
    return text
end 

function textPreprocessing(textFile::String, editfile::String)
    f1 = open(textFile, "r")
    f2 = open(editfile, "w")
    line_text = ""

    while ! eof(f1)
        local line = readline(f1)
        line = strip_accents_and_lowercase(line)
        prev_chr = ""

        if sizeof(line) != 0
            for chr in line
                if chr != "\n"
                    if chr == '.'
                        line_text = line_text * chr * "\n"
                        write(f2, line_text)
                        line_text = ""
                        prev_chr = chr
                    else
                        # in order each sentence to start without whitespace
                        if (prev_chr * chr) == ". "
                            continue
                        else
                            line_text = line_text * chr
                        end
                        prev_chr = chr
                    end
                end
            end 
        end 
    end

    close(f2)
    close(f1)
end

# create input_ids and attention_masks
function inputs(text::String, tokenizer::PyObject)
    f = open(text, "r")
    input_ids = []
    attention_masks = []
    count = 1
    MAX_LEN = 0
    ind = 0
    while ! eof(f)
        sentence = readline(f)

        fromTokenizer = tokenizer(sentence)

        if MAX_LEN < length(fromTokenizer.input_ids)
            MAX_LEN = length(fromTokenizer.input_ids)
            ind = fromTokenizer.input_ids
        end
        
        if count != 1
            temp1 = [fromTokenizer.input_ids]
            input_ids = [input_ids; temp1]
            temp2 = [fromTokenizer.attention_mask]
            attention_masks = [attention_masks; temp2]
        else
            push!(input_ids,fromTokenizer.input_ids)
            push!(attention_masks,fromTokenizer.attention_mask)
        end
        count += 1
    end
    close(f)


    # println(MAX_LEN)
    # println(ind)

    for seq in eachindex(input_ids)
        if length(input_ids[seq]) != MAX_LEN
            for i in 1:(MAX_LEN - length(input_ids[seq]))
                push!(input_ids[seq],tokenizer.pad_token_id)
                push!(attention_masks[seq],0)
            end
        end
    end

    return input_ids, attention_masks
end

end
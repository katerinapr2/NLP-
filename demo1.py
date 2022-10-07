import torch
from transformers import AutoTokenizer, AutoModel
from keras.preprocessing.sequence import pad_sequences
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load pre-trained model tokenizer (vocabulary)
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
config = AutoModel.from_pretrained("nlpaueb/bert-base-greek-uncased-v1", output_hidden_states=True)
model = AutoModel.from_pretrained("nlpaueb/bert-base-greek-uncased-v1", config=config)


sentences = "χτυπησα το ποδι μου στο ποδι του τραπεζιου"

marked_text = "[CLS] " + sentences + " [SEP]"

tokenized = tokenizer.tokenize(marked_text)
input_ids = tokenizer.convert_tokens_to_ids(tokenized)

print(input_ids)

# Print sentence 0, now as a list of IDs.
print('Original: ', sentences)
print('Token IDs:', input_ids)

segments_ids = [1] * len(input_ids)

print (segments_ids)

# Convert to tensors.
inputs_tensor = torch.tensor([input_ids])
att_tensor = torch.tensor([segments_ids])

print(input_ids)
print(att_tensor)

# Predict hidden states features for each layer
with torch.no_grad():
    # all_hidden_states = model(inputs_tensor,att_tensor)[2]
    final_hidden_states = model(inputs_tensor,att_tensor)[0]


layer_i = 0

print ("Number of tokens:", len(final_hidden_states[layer_i]))
batch_i = 0

print ("Number of hidden units:", len(final_hidden_states[layer_i][batch_i]))


import pandas as pd

pca = PCA(n_components=2)
pca = pca.fit(final_hidden_states[0])
pca_transformed = pd.DataFrame(pca.transform(final_hidden_states[0]))
# print(pca_transformed.loc[:,0])
plt.scatter(pca_transformed.loc[:, 0], pca_transformed.loc[:, 1], )

for i in range(final_hidden_states[0].shape[0]):
    plt.text(x=pca_transformed.loc[i,0]+0.3,y=pca_transformed.loc[i,1]+0.3, s=tokenized[i])
    

plt.show()

from transformers import AutoTokenizer, AutoModel
import torch
import csv

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
model = AutoModel.from_pretrained("nlpaueb/bert-base-greek-uncased-v1",  output_hidden_states = True)


# print(tokenizer.vocab)  # {'αφηνεται': 30122, 'κτεο': 30756, 'ταχυτητα': 2082, 'υποστηριξου': 18032, 'αναμενεται': 1334,...}

idx = torch.arange(0,35000)

word_embeddings = model.embeddings.word_embeddings(idx)   #embeddings_lookup_table  [35000 x 768]

with open("./txt_files/embeddings_lookup_table.txt", "w") as embs:
  for i in zip(tokenizer.vocab.keys(), word_embeddings):
    embs.write("%s: %s\n" %(i[0], i[1]))

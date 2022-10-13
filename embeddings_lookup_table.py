from lib2to3.pgen2 import token
from transformers import AutoTokenizer, AutoModel
import torch
import csv

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
model = AutoModel.from_pretrained("nlpaueb/bert-base-greek-uncased-v1",  output_hidden_states = True)


# print(tokenizer.vocab)  # {'αφηνεται': 30122, 'κτεο': 30756, 'ταχυτητα': 2082, 'υποστηριξου': 18032, 'αναμενεται': 1334,...}

word_embeddings = model.embeddings.word_embeddings  #embeddings_lookup_table  [35000 x 768]

with open("./txt_files/embeddings_lookup_table.txt", "w") as embs:
  for key, value in tokenizer.vocab.items(): 
    embs.write("%s: %s\n" %(key, word_embeddings(torch.tensor(value))))

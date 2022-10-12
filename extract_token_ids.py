from pyexpat import model
from transformers import AutoTokenizer, AutoModel

# Load pre-trained model tokenizer (vocabulary)
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")
model = AutoModel.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")


with open("token_ids.txt", 'w') as f: 
    for key, value in tokenizer.vocab.items(): 
        f.write('%s:%s\n' % (key, value))
    
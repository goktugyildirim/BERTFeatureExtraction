#!/usr/bin/env python
# coding: utf-8

# ### BERT Feature Extraction

# In[81]:


input = "Love is between girl and boy. Love is so fantastic. Engineers are very kind people. Engineers are sad."

def BERTFeatureExtraction(input, max_token_size):
    input = input.split(".")
    input = input[:-1]
    
    
    from transformers import BertModel, BertTokenizer
    import torch
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    features = []
    
    for sentence in (input):
        tokens = tokenizer.tokenize(sentence)
        special_tokens = ['[CLS]'] + tokens + ['[SEP]']
        padded_tokens = special_tokens + ['[PAD]' for i in range(max_token_size-len(special_tokens))]
        token_ids = tokenizer.convert_tokens_to_ids(padded_tokens)
        attn_mask = [1 if tok != '[PAD]' else 0 for tok in padded_tokens]
        
        token_ids = torch.tensor(token_ids).unsqueeze(0)
        attn_mask = torch.tensor(attn_mask).unsqueeze(0)
        
        hidden_repr, cls_head = model(token_ids, attention_mask = attn_mask)     
        features.append(hidden_repr[0]) #batch dimension reduction (1, token_size, wv) => (token_size, wv)
    
    import numpy as np
    features = [value.detach().numpy() for value in features]
    features = np.array(features)
    
    return features


max_token_size = 50

features = BERTFeatureExtraction(input, max_token_size) #input is string, output is (batch_size=num_sen, token_size, hidden_dim)
print(features.shape)


# ### Power of the BERT

# In[83]:


from sklearn.cluster import KMeans

features = features.reshape((4,50*768))

model = KMeans(n_clusters=2)
labels = model.fit_predict(features)
labels


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





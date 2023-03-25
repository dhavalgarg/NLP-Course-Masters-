# %%
# !pip install sklearn_crfsuite

# %%
from itertools import chain
import pandas as pd
import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
# import nltk


# %% [markdown]
# ## Read and preprocess the data
# We strongly suggest that you can add pos tags use nltk.pos_tag() function. You can find more information about pos tags in the following link: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
# 
# 
# 

# %%
# Read the data
def read_data(filename):
    rows = []
    with open(f'./ner/wnut16/{filename}') as f:
        for line in f.readlines():
            if len(line) < 2:
                continue
            rows.append(line.rstrip('\n').split())
    data = pd.DataFrame(rows, columns=['term', 'entitytags'])
    # add the pos tags to the dataframe
    # some lines of codes
    # print(data.head())
    tokens=[]
    for word in data['term']:
        tokens.append(word)
    tokens2=nltk.pos_tag(tokens)
    # print(data.tail())
    # print(tokens2)
    tokens3=[]
    for x in tokens2:
            tokens3.append(x[1])
    data['pos'] = tokens3
    # print(data.tail())
    # print(tokens2)
    return data


# %%
train = read_data('train')
test = read_data('test')
dev = read_data('dev')


# %%
# process to get the train, test, dev dataset for crf

def process_data(data):
    dataset = []
    sent = []
    for i, (term, entitytags,pos) in data.iterrows():
        if term == '.':
            sent.append((term, entitytags,pos))
            dataset.append(sent)
            sent = []
        else:
            sent.append((term, entitytags,pos))
    return dataset


# %%
train_sents = process_data(train)
test_sents = process_data(test)
dev_sents = process_data(dev)


# %% [markdown]
# ## The following function will design the feature for crf model. 
# You need to add additional features to this function. The potential features you can add are:
# 1. The characters of the word
# 2. The pos tag of the word
# 3. the word before and after the current word
# 
# There will also be other features that you can add to improve the performance of the model.
# 

# %%
import numpy as np
import gensim
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_input_file = 'src/glove.6B/glove.6B.300d.txt'  # path to the GloVe model file
word2vec_output_file = 'src/glove.6B/glove.6B.300d.word2vec'  # path to the output word2vec model file

# glove2word2vec(glove_input_file, word2vec_output_file)
# Load vectors directly from the file
print("***")
model1 = gensim.models.KeyedVectors.load_word2vec_format('src/glove.6B/glove.6B.300d.word2vec',binary=False) ### Loading pre-trainned word2vec model
print("***")
### Embedding function 
def get_features(word):
    word=word.lower()
    try:
         vector=model1[word]
    except:
        # if the word is not in vocabulary,
        # returns zeros array
        vector=np.zeros(300,)

    return vector
def word2features(sent, i):
    word = sent[i][0]
    emd = get_features(word)
    """
    
    Here we have already provided you some of the features. Without any modification, you can run the code and get the baseline result.
    However, the performance is not good. We suggest you to add more features to improve the performance.
    For example, you can add the character level features, the word shape features, the word embedding features, etc.
    We also suggest you to add the features of the previous word and the next word.
    
    We strongly suggests you to add pos tags as features. You can use the pos tags provided by nltk.pos_tag() to get the pos tags of the words.
    
    """
    prev_word = sent[i-1][0]
    vowels = [word.count(x) for x in "aeiouAEIOU"]
    # next_word = sent[i+1][0]

    # Load pre-trained word embeddingg

    features = {
        'word.lower()': word.lower(),
        # # add more features here
#         'pos_tags': sent[i][2],
#         'word_len': len(word),
#         'capital_start': word[0].isupper(),
#         'count_vowels': sum(vowels),
#         'word.isdigit': word.isdigit(),
#         'word.istitle': word.istitle(),
#         # 'current_word_title':word[0].title(),
#         'prev_word_len':len(prev_word),
#         'prev_word.lower':prev_word.lower(),
#         # 'word_emb': emb,
#         # 'prev_word_emb': prev_embedding,
#         'prev_word_tag':sent[i-1][2],
#         'prev_word_capital':prev_word[0].isupper(),
#         'prev_word.istitle':prev_word.istitle(),
#         'prev_word.isdigit': prev_word.isdigit()
#         # 'next_word.isdigit':next_word.isdigit(),
#         # 'next_word.istitle':next_word.istitle()
    }
    # for iv,value in enumerate(emd):
#         features['word_emd{}'.format(iv)]=value
#     
#     if i < len(sent)-1:
#         next_word = sent[i+1][0]
#         features.update({
#             'next_word.isdigit':next_word.isdigit(),
#             'next_word.istitle':next_word.istitle()
#         })

    return features
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, label,pos in sent]


def sent2tokens(sent):
    return [token for token, label,pos in sent]


# %%
# sent2features(train_sents[0])[0]


# %%
X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

X_dev = [sent2features(s) for s in dev_sents]
y_dev = [sent2labels(s) for s in dev_sents]
print("***")


# %% [markdown]
# The following block of code will help you visualize the feature for a given sentence.

# %%
# X_train[0]

# %% [markdown]
# In the following block of code, we use try and except because the version of the library.

# %%

crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
try:
    crf.fit(X_train, y_train)
except AttributeError:
    pass

# %% [markdown]
# This block of code will help you visualize the learned features for crf model.

# %%
labels = list(crf.classes_)
labels.remove('O')
# labels


# %%
y_pred = crf.predict(X_test)
print(metrics.flat_f1_score(y_test, y_pred,
                      average='weighted', labels=labels))


# %%
words = [sent2tokens(s) for s in test_sents]


# %%
labels = [sent2labels(s) for s in test_sents]


# %%
predictions = []
for (word, true_id, pred_id) in zip(words, labels, y_pred):
    for (w, t, p) in zip(word, true_id, pred_id):
        line = ' '.join([w, t, p])
        predictions.append(line)
    predictions.append('')
with open('crf_pred', 'w') as f:
    f.write('\n'.join(predictions))
          

# %%
import os
eval_script = '../released/src/conlleval'
predf = 'crf_pred'
scoref = 'crf_score'
os.system('%s < %s > %s' % (eval_script, predf, scoref))


# %%
eval_lines = [l.rstrip() for l in open(scoref, 'r', encoding='utf8')]

for i, line in enumerate(eval_lines):
    print(line)

# %% [markdown]
# ## Let's check what classifier learned:
# 
# You will need to analyze how the transition the model is learned in the report.

# %%
from collections import Counter

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

print("Top likely transitions:")
print_transitions(Counter(crf.transition_features_).most_common(20))

print("\nTop unlikely transitions:")
print_transitions(Counter(crf.transition_features_).most_common()[-20:])

# %% [markdown]
# ## Check the state features:
# 
# You will need to analyze how your features will help the model to learn the correct labels.

# %%
def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))    

print("Top positive:")
print_state_features(Counter(crf.state_features_).most_common(30))

print("\nTop negative:")
print_state_features(Counter(crf.state_features_).most_common()[-30:])



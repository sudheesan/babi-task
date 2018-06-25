from __future__ import print_function
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from functools import reduce
import numpy as np
import re

def tokenize(sent):
    
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
 
    data = []
    story = []
    for line in lines:
        line = line.strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data

def parse_input_story(input_story,question):
    return [(tokenize(input_story),tokenize(question))]
    
    

def get_stories(f, only_supporting=False, max_length=None):

    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    for story, query in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        X.append(x)
        Xq.append(xq)
    return (pad_sequences(X, maxlen=story_maxlen),
            pad_sequences(Xq, maxlen=query_maxlen))




train_stories = get_stories(open('tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt','r'))
test_stories = get_stories(open('tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_test.txt','r'))


vocab = set()
for story, q, answer in train_stories + test_stories:
    vocab |= set(story + q + [answer])
vocab = sorted(vocab)

vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

json_file = open('model_1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model1.h5")

def getAnswer(parsed_stories):
    inputs_train, queries_train = vectorize_stories(parsed_stories,word_idx,story_maxlen,query_maxlen)
    pred_results = loaded_model.predict(([inputs_train, queries_train]))
    val_max = np.argmax(pred_results[0])
    for key, val in word_idx.items():
        if val == val_max:
            k = key
    return k
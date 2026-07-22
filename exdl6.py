import numpy as np
import pandas as pd
import json
import functools as fc
from sklearn.metrics import accuracy_score
train = pd.read_csv('data1/train', sep='\t', names=['index', 'word', 'POS'])
index = train['index'].tolist()
word = train['word'].tolist()
pos = train['POS'].tolist()
vocab = {}
for w in word:
    vocab[w] = vocab.get(w, 0) + 1
vocab2 = {}
num_unk = 0
for w, c in vocab.items():
    if c >= 3:
        vocab2[w] = c
    else:
        num_unk += c
vocab_sorted = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
with open("output/vocab_frequent.txt", "w") as f:
    f.write(f"<unk>\t0\t{num_unk}\n")
    for i, (w, c) in enumerate(vocab_sorted):
        f.write(f"{w}\t{i+1}\t{c}\n")
print("The total size of my vocabulary is", len(vocab_sorted))
print("The total occurrences of <unk> is", num_unk)
vocab_ls = list(vocab2.keys())
with open("output/vocab_words.txt", "w") as f:
    for w in vocab_ls:
        f.write(w + "\n")
word = [w if w in vocab_ls else "<unk>" for w in word]
ss = {}
sx = {}
for i in range(len(word) - 1):
    if index[i] < index[i + 1]:
        ss[pos[i + 1] + "|" + pos[i]] = ss.get(pos[i + 1] + "|" + pos[i], 0) + 1
        sx[word[i] + "|" + pos[i]] = sx.get(word[i] + "|" + pos[i], 0) + 1
for i in range(len(index)):
    if index[i] == 1:
        ss[pos[i] + "|<s>"] = ss.get(pos[i] + "|<s>", 0) + 1
count_pos = {}
for p in pos:
    count_pos[p] = count_pos.get(p, 0) + 1
count_pos["<s>"] = index.count(1)
emission = {}
transition = {}
for k, v in sx.items():
    emission[k] = v / count_pos[k.split("|")[1]]
for k, v in ss.items():
    transition[k] = v / count_pos[k.split("|")[1]]
print("There are", len(transition), "transition parameters in my HMM")
print("There are", len(emission), "emission parameters in my HMM")
with open("output/hmm.json", "w") as f:
    json.dump([emission, transition], f)
pos_distinct = list(count_pos.keys())
with open("output/pos.txt", "w") as f:
    f.write("\n".join(pos_distinct))
def load_sentences(file):
    data = pd.read_csv(file, sep='\t', names=['index', 'word', 'POS'])
    idx = data['index'].tolist()
    words = data['word'].tolist()
    tags = data['POS'].tolist()
    sent_words, sent_tags = [], []
    w, p = [], []
    for i in range(len(data) - 1):
        w.append(words[i])
        p.append(tags[i])
        if idx[i] >= idx[i + 1]:
            sent_words.append(w)
            sent_tags.append(p)
            w, p = [], []
    w.append(words[-1])
    p.append(tags[-1])
    sent_words.append(w)
    sent_tags.append(p)
    return sent_words, sent_tags
with open("output/vocab_words.txt") as f:
    vocab_frequent = [x.strip() for x in f]
with open("output/pos.txt") as f:
    pos_distinct = [x.strip() for x in f]
with open("output/hmm.json") as f:
    emission, transition = json.load(f)
word_dev2, pos_dev2 = load_sentences("data1/dev")
def greedy(sentence):
    pred = []
    for i, w in enumerate(sentence):
        if w not in vocab_frequent:
            w = "<unk>"
        best_pos = "UNK"
        best_prob = 0
        for p in pos_distinct:
            prev = "<s>" if i == 0 else pred[-1]

            prob = emission.get(w + "|" + p, 0) * transition.get(p + "|" + prev, 0)

            if prob > best_prob:
                best_prob = prob
                best_pos = p
        pred.append(best_pos)
    return pred
greedy_pred = fc.reduce(lambda a, b: a + b,
                        [greedy(s) for s in word_dev2])
actual = fc.reduce(lambda a, b: a + b, pos_dev2)
print("The prediction accuracy on the dev data is {:.2f}%".format(
    accuracy_score(actual, greedy_pred) * 100
))
def viterbi(sentence):
    V = [{}]
    path = {}
    for p in pos_distinct:
        w = sentence[0] if sentence[0] in vocab_frequent else "<unk>"
        V[0][p] = emission.get(w + "|" + p, 0) * transition.get(p + "|<s>", 0)
        path[p] = [p]
    for t in range(1, len(sentence)):
        V.append({})
        new_path = {}
        w = sentence[t] if sentence[t] in vocab_frequent else "<unk>"
        for p in pos_distinct:
            best_prob = -1
            best_state = None
            for prev in pos_distinct:
                prob = V[t - 1][prev] * \
                       transition.get(p + "|" + prev, 0) * \
                       emission.get(w + "|" + p, 0)
                if prob > best_prob:
                    best_prob = prob
                    best_state = prev
            V[t][p] = best_prob
            new_path[p] = path[best_state] + [p]
        path = new_path
    last = max(pos_distinct, key=lambda x: V[-1][x])
    return path[last]
viterbi_pred = fc.reduce(lambda a, b: a + b,
                         [viterbi(s) for s in word_dev2])
print("The prediction accuracy on the dev data using Viterbi is {:.2f}%".format(
    accuracy_score(actual, viterbi_pred) * 100
))

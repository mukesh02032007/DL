import numpy as np
import pandas as pd
import json
import functools as fc
from sklearn.metrics import accuracy_score
# Task 1: Vocabulary Creation
# train = pd.read_csv('data1/train', sep='\t', names=['index', 'word', 'POS'])
train = pd.read_csv('data1/train', sep='\t', names=['index', 'word', 'POS'])
train.head()
word = train['word'].values.tolist()
index = train['index'].values.tolist()
pos = train['POS'].values.tolist()
vocab = {}
for i in range(len(word)):
 if word[i] in vocab:
 vocab[word[i]] += 1
 else:
 vocab[word[i]] = 1
# Replace rare words with <unk> (threshold=3)
vocab2 = {}
num_unk = 0
for w in vocab:
 if vocab[w] >= 3:
 vocab2[w] = vocab[w]
 else:
 num_unk += vocab[w]
# Sort the vocabulary by occurrences of words
vocab_sorted = sorted(vocab.items(), key=lambda item: item[1], reverse=True)
# Write the sorted vocabulary to a file (format: word index occurrence)
with open('output/vocab_frequent.txt', 'w') as vocab_file:
 # Add <unk> to the top of the vocabulary manually
 vocab_file.write('<unk>\t0\t' + str(num_unk) + '\n')
 for i in range(len(vocab_sorted)):
 vocab_file.write(vocab_sorted[i][0] + '\t' + str(i+1) + '\t' + str(vocab_sorted[i][1]) + '\n')
print(f'The total size of my vocabulary is {len(vocab_sorted)}\n')
print(f'The total occurrences of <unk> is {num_unk}\n')
# Task 2: Model Learning
# Build a vocabulary list with only frequent words (occurring at least 3 times)
vocab_ls = list(vocab2.keys())
# Write the frequent words into a file
with open('output/vocab_frequent.txt', 'w') as output:
 for word in vocab_ls:
 output.write(word + '\n')
# Replace words not in vocab_ls with <unk>
for i in range(len(word)):
 if word[i] not in vocab_ls:
 word[i] = '<unk>'
# Count (s, s') and (s, x) pairs
ss = {} # transition counts: pos[i+1] | pos[i]
sx = {} # emission counts: word[i] | pos[i]
for i in range(len(word)-1):
 # Ensure the index of current word is less than next word (i.e. same sentence)
 if index[i] < index[i+1]:
 key_ss = str(pos[i+1]) + '|' + str(pos[i])
 if key_ss in ss:
 ss[key_ss] += 1
 else:
 ss[key_ss] = 1
 key_sx = str(word[i]) + '|' + str(pos[i])
 if key_sx in sx:
 sx[key_sx] += 1
 else:
 sx[key_sx] = 1
# Count occurrences of POS at beginning of sequences (s | <s>)
for i in range(len(word)):
 if index[i] == 1:
 key_start = str(pos[i]) + '|' + '<s>'
 if key_start in ss:
 ss[key_start] += 1
 else:
 ss[key_start] = 1
# Build emission and transition dictionaries
emission = {}
transition = {}
# Count occurrences of POS tags
count_pos = {}
for p in pos:
 if p in count_pos:
 count_pos[p] += 1
 else:
 count_pos[p] = 1
# Don't forget to count <s> (start token)
count_pos['<s>'] = 0
for i in range(len(index)):
 if index[i] == 1:
 count_pos['<s>'] += 1
# Calculate emission probabilities: emission[(s, x)] = count(s, x) / count(s)
for sx_pair in sx:
 pos_tag = sx_pair.split('|')[1]
 emission[sx_pair] = sx[sx_pair] / count_pos[pos_tag]
# Calculate transition probabilities: transition[(s, s')] = count(s, s') / count(s)
for ss_pair in ss:
 pos_tag = ss_pair.split('|')[1]
 transition[ss_pair] = ss[ss_pair] / count_pos[pos_tag]
print(f'There are {len(transition)} transition parameters in my HMM\n')
print(f'There are {len(emission)} emission parameters in my HMM\n')
# Write the emission and transition dictionaries into a JSON file
emission_transition = [emission, transition]
with open('output/hmm.json', 'w') as output:
 json.dump(emission_transition, output)
# Build a list of distinct POS
pos_distinct = list(count_pos.keys())
# Write the pos_distinct into a txt file
with open('output/pos.txt', 'w') as pos_output:
 for pos_tag in pos_distinct:
 pos_output.write(pos_tag + '\n')
# Task 3: Greedy Decoding with HMM
# Load vocab frequent words
vocab_frequent = []
with open('output/vocab_frequent.txt', 'r') as vocab_txt:
 for word in vocab_txt:
 vocab_frequent.append(word.strip('\n'))
# Load POS tags
pos_distinct = []
with open('output/pos.txt', 'r') as pos_txt:
 for pos_tag in pos_txt:
 pos_distinct.append(pos_tag.strip('\n'))
with open('output/hmm.json', 'r') as hmm:
 json_data = json.load(hmm)
emission, transition = json_data[0], json_data[1]
# Load dev dataset
# dev = pd.read_csv('data1/dev', sep='\t', names=['index', 'word', 'POS'])
dev = pd.read_csv('data1/dev', sep='\t', names=['index', 'word', 'POS'])
index_dev = dev['index'].values.tolist()
word_dev = dev['word'].values.tolist()
pos_dev = dev['POS'].values.tolist()
# Split dev lists (index, word, pos) into individual sentences (list of lists)
word_dev2 = []
pos_dev2 = []
word_sample = []
pos_sample = []
for i in range(len(dev)-1):
 word_sample.append(word_dev[i])
 pos_sample.append(pos_dev[i])
 if index_dev[i] >= index_dev[i+1]:
 word_dev2.append(word_sample)
 word_sample = []
 pos_dev2.append(pos_sample)
 pos_sample = []
# Append last sample
word_sample.append(word_dev[-1])
pos_sample.append(pos_dev[-1])
word_dev2.append(word_sample)
pos_dev2.append(pos_sample)
def greedy(sentence):
 pos = []
 # Make sure first word is in vocab, else <unk>
 if sentence[0] not in vocab_frequent:
 sentence[0] = '<unk>'
 # Predict POS of first word based on max emission*transition
 max_prob = 0
 p0 = 'UNK'
 for p in pos_distinct:
 try:
 temp = emission[sentence[0] + '|' + p] * transition[p + '|' + '<s>']
 if temp > max_prob:
 max_prob = temp
 p0 = p
 except:
 pass
 pos.append(p0)
 if sentence[i] not in vocab_frequent:
 sentence[i] = '<unk>'
 max_prob = 0
 pi = 'UNK'
 for p in pos_distinct:
 try:
 temp = emission[sentence[i] + '|' + p] * transition[p + '|' + pos[-1]]
 if temp > max_prob:
 max_prob = temp
 pi = p
 except:
 pass
 pos.append(pi)
 return pos
pos_greedy = [greedy(s) for s in word_dev2]
# Flatten the list of lists into a single list
pos_greedy = fc.reduce(lambda a, b: a + b, pos_greedy)
pos_dev_flat = fc.reduce(lambda a, b: a + b, pos_dev2)
acc = accuracy_score(pos_dev_flat, pos_greedy)
print('The prediction accuracy on the dev data is {:.2f}%'.format(acc * 100))
# Task 4: Viterbi Decoding with HMM
# Load vocab frequent words
vocab_frequent = []
with open('output/vocab_frequent.txt', 'r') as vocab_txt:
 for word in vocab_txt:
 vocab_frequent.append(word.strip('\n'))
# Load POS tags
pos_distinct = []
with open('output/pos.txt', 'r') as pos_txt:
 for pos_tag in pos_txt:
 pos_distinct.append(pos_tag.strip('\n'))
# Load emission and transition dictionaries
with open('output/hmm.json', 'r') as hmm:
 json_data = json.load(hmm)
emission, transition = json_data[0], json_data[1]
# Load dev dataset
dev = pd.read_csv('data1/dev', sep='\t', names=['index', 'word', 'POS'])
index_dev = dev['index'].values.tolist()
word_dev = dev['word'].values.tolist()
pos_dev = dev['POS'].values.tolist()
# Split dev lists into sentences
word_dev2 = []
pos_dev2 = []
word_sample = []
pos_sample = []
for i in range(len(dev) - 1):
 word_sample.append(word_dev[i])
 pos_sample.append(pos_dev[i])
 if index_dev[i] >= index_dev[i + 1]:
 word_dev2.append(word_sample)
 word_sample = []
 pos_dev2.append(pos_sample)
 pos_sample = []
# Append last sample
word_sample.append(word_dev[-1])
pos_sample.append(pos_dev[-1])
word_dev2.append(word_sample)
pos_dev2.append(pos_sample)
def viterbi(sentence):
 path = {}
 V = [{}]
 # Initialization step
 for p in pos_distinct:
 if sentence[0] not in vocab_frequent:
 sentence[0] = '<unk>'
 try:
 V[0][p] = emission[sentence[0] + '|' + p] * transition[p + '|' + '<s>']
 except KeyError:
 V[0][p] = 0
 path[p] = [p]
 # Recursion step
 for t in range(1, len(sentence)):
 V.append({})
 new_path = {}
 if sentence[t] not in vocab_frequent:
 sentence[t] = '<unk>'
 for p in pos_distinct:
 max_prob = 0
 prev_state = None
 for p_prev in pos_distinct:
 prob = V[t-1][p_prev] * transition.get(p + '|' + p_prev, 0) * emission.get(sentence[t] + '|' + 
p,0)
 if prob > max_prob:
 max_prob = prob
 prev_state = p_prev
 V[t][p] = max_prob
 new_path[p] = path[prev_state] + [p]
 path = new_path
  max_prob = 0
 best_path = None
 for p in pos_distinct:
 if V[-1][p] > max_prob:
 max_prob = V[-1][p]
 best_path = path[p]
 return best_path
pos_viterbi = [viterbi(s) for s in word_dev2]
pos_viterbi_flat = fc.reduce(lambda a, b: a + b, pos_viterbi)
pos_dev_flat = fc.reduce(lambda a, b: a + b, pos_dev2)
acc_viterbi = accuracy_score(pos_dev_flat, pos_viterbi_flat)
print('The prediction accuracy on the dev data using Viterbi is {:.2f}%'.format(acc_viterbi * 100))
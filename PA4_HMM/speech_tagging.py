import numpy as np
import time
import random
from hmm import HMM
from collections import defaultdict, Counter




def accuracy(predict_tagging, true_tagging):
    if len(predict_tagging) != len(true_tagging):
        return 0, 0, 0
    cnt = 0
    for i in range(len(predict_tagging)):
        if predict_tagging[i] == true_tagging[i]:
            cnt += 1
    total_correct = cnt
    total_words = len(predict_tagging)
    if total_words == 0:
        return 0, 0, 0
    return total_correct, total_words, total_correct * 1.0 / total_words


class Dataset:

    def __init__(self, tagfile, datafile, train_test_split=0.8, seed=int(time.time())):
        tags = self.read_tags(tagfile)
        data = self.read_data(datafile)
        self.tags = tags
        lines = []
        for l in data:
            new_line = self.Line(l)
            if new_line.length > 0:
                lines.append(new_line)
        if seed is not None: np.random.seed(seed)
        np.random.shuffle(lines)
        train_size = int(train_test_split * len(data))
        self.train_data = lines[:train_size]
        self.test_data = lines[train_size:]
        return

    def read_data(self, filename):
        with open(filename, 'r') as f:
            sentence_lines = f.read().split("\n\n")
        return sentence_lines

    def read_tags(self, filename):
        with open(filename, 'r') as f:
            tags = f.read().split("\n")
        return tags

    class Line:
        def __init__(self, line):
            words = line.split("\n")
            self.id = words[0]
            self.words = []
            self.tags = []

            for idx in range(1, len(words)):
                pair = words[idx].split("\t")
                self.words.append(pair[0])
                self.tags.append(pair[1])
            self.length = len(self.words)
            return

        def show(self):
            print(self.id)
            print(self.length)
            print(self.words)
            print(self.tags)
            return


# TODO:



def model_training(train_data, tags):
    from collections import defaultdict, Counter
    model = None

    ###################################################
    def normalize_states(arr, axis=None):
        if axis is None:
            axis='1'
        S = arr.shape[0]
        if len(arr.shape) == 1:
            cumsum = sum(arr)
            temp = arr / cumsum
            return temp
        O = arr.shape[1]
        normalized = np.zeros((S, O))
        if axis==1:
            for i in np.arange(O): #sum over rows
                cumsum = np.sum(arr[i])
                normalized[i] = arr[i] / cumsum
        else:
            for i in np.arange(S):
                cumsum= np.sum(arr[:,1])
                normalized[i] = arr[i] / cumsum

        return normalized

    #def tree():
    #    return defaultdict(tree)

    num_states = len(tags)

    pi = np.zeros(num_states)
    A = np.zeros((num_states, num_states))

    obs_dict = {}
    state_dict = {pos: tags.index(pos) for pos in tags}
    state_dict['end']=len(tags)

    # get probabilities for model parameters

    #transitionCounter = Counter()
    transitionCounter = defaultdict(lambda: defaultdict(int))

    piCounter = defaultdict(int)
    #piCounter = Counter()

    state_obs_counts =defaultdict(lambda: defaultdict(int))
    # =defaultdict(Counter)

    #transitions=defaultdict(Counter)

    obs_state_dict = {}
    wordcount = 0
    obs_words = []
    for line in train_data:
        #transitionCounter.update((zip(line.tags, line.tags[1:])))
        #piCounter.update([line.tags[0]])
        piCounter[line.tags[0]]+=1
        pos=line.tags+['end']
        for i in np.arange(line.length):
            if obs_state_dict.get(line.words[i],None) is None:
            #if line.words[i] not in obs_words:
                obs_words.append(line.words[i])
                obs_state_dict[line.words[i]] = line.tags[i]
            transitionCounter[state_dict[pos[i]]][state_dict[pos[i+1]]]+=1
            state_obs_counts[state_dict[line.tags[i]]][line.words[i]] += 1
            wordcount += 1
    #dictA = {(state_dict[k[0]], state_dict[k[1]]): v for k, v in transitionCounter.items()}
    dictPi = {state_dict[k]: v / sum(piCounter.values()) for k, v in piCounter.items()}
    for i in np.arange(num_states):
        pi[i] = dictPi.get(i, 10 ** -6)
        for j in np.arange(num_states):
            #A[i][j] = dictA.get((i, j), 10 ** -6)
            A[i][j] = transitionCounter.get(i, {}).get(j, 10**-6)
            #A[i][j] = dictPi.get(i, 10 ** -6)*dictPi.get(j,10**-6)
    A = normalize_states(A)
    B = np.zeros((num_states, wordcount + 1))
    for tag_idx, words in state_obs_counts.items():
        cum_count = sum(state_obs_counts[tag_idx].values())
        for word, cnt in words.items():
            obs_dict[word] = obs_words.index(word)
            B[tag_idx, obs_dict[word]] = cnt / cum_count

    B[:, wordcount] = 10 ** -6
    model = HMM(pi, A, B, obs_dict, state_dict)
    model.num_words = wordcount
    model.unknown_state = wordcount
    #model.obs = obs_words.append(None)
    model.obs_state_dict = obs_state_dict
    print("obs_state_dict", obs_state_dict)
    model.num_states = num_states
    ###################################################
    return model

# TODO:

def speech_tagging(test_data, model, tags):
   
    tagging = []
    ###################################################
    for line in test_data:
        seq = []
        tag_seq = []
        for i, word in enumerate(line.words):
            seq.append(word)
            if model.obs_state_dict.get(word, None) is None:
               
                model.obs_dict[word] = model.unknown_state
                predicted_path = model.viterbi(np.array(seq))
                predicted_state = predicted_path[-1]
                model.update_emissions(model.state_dict[predicted_state])
                model.obs_dict[word] = model.num_words + 1
                model.obs_state_dict[word] = predicted_state

                model.num_words += 1
                tag_seq.append(predicted_state)

            else:
                tag_seq.append(model.obs_state_dict[word])
            
        tagging.append(tag_seq)

    ###################################################
    return tagging

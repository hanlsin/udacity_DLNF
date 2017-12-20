def pretty_print_review_and_label(i):
    print(labels[i] + '\t:\t' + reviews[i][:80] + '...')


g = open('/Users/yp/study/udacity/DLNF/udacity_DLNF/Part1.NeuralNetworks/L7.SentimentAnalysis/reviews.txt', 'r')
reviews = list(map(lambda x: x[:-1], g.readlines()))
g.close()

g = open('/Users/yp/study/udacity/DLNF/udacity_DLNF/Part1.NeuralNetworks/L7.SentimentAnalysis/labels.txt', 'r')
labels = list(map(lambda x: x[:-1].upper(), g.readlines()))
g.close()

assert(len(reviews) == 25000)
pretty_print_review_and_label(0)
pretty_print_review_and_label(4998)

'''
from collections import Counter
import numpy as np

pos_cnts = Counter()
neg_cnts = Counter()
tot_cnts = Counter()
for i, l in enumerate(labels):
    if l == 'POSITIVE':
        for w in reviews[i].split(' '):
            pos_cnts[w] += 1
            tot_cnts[w] += 1
    else:
        for w in reviews[i].split(' '):
            neg_cnts[w] += 1
            tot_cnts[w] += 1
print('POSITIVE WORDS\n', pos_cnts.most_common(3))
print('NEGATIVE WORDS\n', neg_cnts.most_common(3))
print('TOTAL WORDS\n', tot_cnts.most_common(3))

pos_neg_ratios = Counter()
for w, cnt in list(tot_cnts.most_common()):
    if cnt > 100:
        pos_neg_ratios[w] = pos_cnts[w] / float(neg_cnts[w] + 1)

print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
print(
    "Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))


for w, _ in list(pos_neg_ratios.items()):
    pos_neg_ratios[w] = np.log(pos_neg_ratios[w])

print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
print(
    "Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))
'''

import time
import sys
import numpy as np


class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_node_cnt=10, learning_rate=0.1):
        np.random.seed(1)
        self.reviews = reviews
        self.labels = labels
        self.preprocess_data(reviews, labels)

        self.input_node_cnt = len(self.reviews_vocab)
        self.hidden_node_cnt = hidden_node_cnt
        self.output_node_cnt = 1
        self.learning_rate = learning_rate

        self.init_network()

    def preprocess_data(self, reviews, labels):
        reviews_vocab = set()
        for r in reviews:
            for w in r.split(' '):
                reviews_vocab.add(w)
        self.reviews_vocab = list(reviews_vocab)

        self.word2index = {}
        for i, w in enumerate(self.reviews_vocab):
            self.word2index[w] = i

        labels_vocab = set()
        for l in labels:
            labels_vocab.add(l)
        self.labels_vocab = list(labels_vocab)

        self.label2index = {}
        for i, l in enumerate(self.labels_vocab):
            self.label2index[l] = i

    def init_network(self):
        self.input = np.zeros(shape=(1, self.input_node_cnt))
        self.hidden_output = np.zeros(shape=(1, self.hidden_node_cnt))

        self.w_i_h = np.zeros(
            shape=(self.input_node_cnt, self.hidden_node_cnt))

        self.w_h_o = np.random.normal(
            0.0, self.output_node_cnt**-0.5,
            (self.hidden_node_cnt, self.output_node_cnt))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def update_input_layer(self, review):
        self.input *= 0
        for w in review.split(' '):
            if w in self.word2index.keys():
                self.input[0, self.word2index[w]] = 1

    def train2(self):
        # Keep track of correct predictions to display accuracy during training
        correct_so_far = 0

        last_progress = 0.0

        # Remember when we started for printing time statistics
        start = time.time()

        training_reviews = list()
        for r in self.reviews:
            indices = set()
            for w in r.split(' '):
                if w in self.word2index.keys():
                    indices.add(self.word2index[w])
            training_reviews.append(list(indices))

        sys.stdout.write("\rPrepared Training Set: time spending: " +
                         str(float(time.time() - start)) + " sec.\n")
        print("=========================")

        # Remember when we started for printing time statistics
        start = time.time()

        for i in range(len(training_reviews)):
            reviews = training_reviews[i]
            label = self.labels[i]
            target = 0
            if label == 'POSITIVE':
                target = 1

            # print('I[', self.input.shape, ']:', self.input)
            # print('W I -> H: [', self.w_i_h.shape, ']')
            # print('W H -> O: [', self.w_h_o.shape, ']')
            # Forward pass
            # input layer --> hidden layer
            self.hidden_output *= 0
            for idx in reviews:
                self.hidden_output += self.w_i_h[idx]
            # print('I -> H:', ho, ho.shape)
            # hidden layer --> output layer
            oo = np.dot(self.hidden_output, self.w_h_o)
            # print('H -> O:', oo, oo.shape)
            # activation function
            oo = self.sigmoid(oo)
            # print('AF:', oo, oo.shape)

            # Backward pass
            # output error
            oe = target - oo
            # print('Output Error :', oe, oe.shape)
            # output error term
            oet = oe * oo * (1 - oo)
            # print('Output Error Term :', oet, oet.shape)
            # output delta
            w_o_h_delta = self.learning_rate * \
                np.dot(self.hidden_output.T, oet)
            # print('Output -> Hidden Weights Delta:',
            #       w_o_h_delta, w_o_h_delta.shape)
            # hidden error term
            het = np.dot(oet, self.w_h_o.T)

            # update the weights
            self.w_h_o += w_o_h_delta
            for idx in reviews:
                self.w_i_h[idx] += self.learning_rate * het[0]

            # Keep track of correct predictions.
            if(oo >= 0.5 and target == 1):
                correct_so_far += 1
            elif(oo < 0.5 and target == 0):
                correct_so_far += 1

            now_progress = 100 * i / float(len(self.reviews))
            if (now_progress - last_progress) < 10:
                continue

            last_progress = now_progress

            # For debug purposes, print out our prediction accuracy and speed
            # throughout the training process.
            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(now_progress)[:4]
                             + "% Speed(reviews/sec):" +
                             str(reviews_per_second)[0:5]
                             + " #Correct:" +
                             str(correct_so_far) + " #Trained:" + str(i + 1)
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i + 1))[:4] + "%\n")

    def run2(self, review):
        indices = set()
        for w in review.lower().split(' '):
            if w in self.word2index.keys():
                indices.add(self.word2index[w])

        self.hidden_output *= 0
        for idx in indices:
            self.hidden_output += self.w_i_h[idx]
        oo = self.sigmoid(np.dot(self.hidden_output, self.w_h_o))

        if oo >= 0.5:
            return 'POSITIVE'
        else:
            return 'NEGATIVE'

    def test2(self, testing_reviews, testing_labels):
        correct = 0
        start = time.time()

        for i in range(len(testing_labels)):
            prediction = self.run2(testing_reviews[i])
            if (prediction == testing_labels[i]):
                correct += 1

            now_progress = 100 * i / float(len(testing_reviews))

            # For debug purposes, print out our prediction accuracy and speed
            # throughout the training process.
            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

        sys.stdout.write("\rProgress:" + str(now_progress)[:4]
                         + "% Speed(reviews/sec):" +
                         str(reviews_per_second)[0:5]
                         + " #Correct:" + str(correct)
                         + " #Tested:" + str(i + 1)
                         + " Testing Accuracy:" +
                         str(correct * 100 / float(i + 1))[:4] + "%\n")

    def train(self):
        # Keep track of correct predictions to display accuracy during training
        correct_so_far = 0

        last_progress = 0.0

        # Remember when we started for printing time statistics
        start = time.time()

        for i, r in enumerate(self.reviews):
            label = self.labels[i]
            target = 0
            if label == 'POSITIVE':
                target = 1

            self.update_input_layer(r)

            # print('I[', self.input.shape, ']:', self.input)
            # print('W I -> H: [', self.w_i_h.shape, ']')
            # print('W H -> O: [', self.w_h_o.shape, ']')
            # Forward pass
            # input layer --> hidden layer
            ho = np.dot(self.input, self.w_i_h)
            # print('I -> H:', ho, ho.shape)
            # hidden layer --> output layer
            oo = np.dot(ho, self.w_h_o)
            # print('H -> O:', oo, oo.shape)
            # activation function
            oo = self.sigmoid(oo)
            # print('AF:', oo, oo.shape)

            # Backward pass
            # output error
            oe = target - oo
            # print('Output Error :', oe, oe.shape)
            # output error term
            oet = oe * oo * (1 - oo)
            # print('Output Error Term :', oet, oet.shape)
            # output delta
            w_o_h_delta = self.learning_rate * np.dot(ho.T, oet)
            # print('Output -> Hidden Weights Delta:',
            #       w_o_h_delta, w_o_h_delta.shape)
            # hidden error term
            het = np.dot(oet, self.w_h_o.T)
            w_h_i_delta = self.learning_rate * np.dot(self.input.T, het)
            # print('Hidden -> Input Weights Delta:',
            #       w_h_i_delta, w_h_i_delta.shape)

            # update the weights
            self.w_h_o += w_o_h_delta
            self.w_i_h += w_h_i_delta

            # Keep track of correct predictions.
            if(oo >= 0.5 and target == 1):
                correct_so_far += 1
            elif(oo < 0.5 and target == 0):
                correct_so_far += 1

            now_progress = 100 * i / float(len(self.reviews))
            if (now_progress - last_progress) < 10:
                continue

            last_progress = now_progress

            # For debug purposes, print out our prediction accuracy and speed
            # throughout the training process.
            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(now_progress)[:4]
                             + "% Speed(reviews/sec):" +
                             str(reviews_per_second)[0:5]
                             + " #Correct:" +
                             str(correct_so_far) + " #Trained:" + str(i + 1)
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i + 1))[:4] + "%\n")

    def run(self, review):
        self.update_input_layer(review)

        ho = np.dot(self.input, self.w_i_h)
        oo = self.sigmoid(np.dot(ho, self.w_h_o))

        if oo >= 0.5:
            return 'POSITIVE'
        else:
            return 'NEGATIVE'

    def test(self, testing_reviews, testing_labels):
        correct = 0
        start = time.time()

        for i in range(len(testing_labels)):
            prediction = self.run(testing_reviews[i])
            if (prediction == testing_labels[i]):
                correct += 1

            now_progress = 100 * i / float(len(testing_reviews))

            # For debug purposes, print out our prediction accuracy and speed
            # throughout the training process.
            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

        sys.stdout.write("\rProgress:" + str(now_progress)[:4]
                         + "% Speed(reviews/sec):" +
                         str(reviews_per_second)[0:5]
                         + " #Correct:" + str(correct)
                         + " #Tested:" + str(i + 1)
                         + " Testing Accuracy:" +
                         str(correct * 100 / float(i + 1))[:4] + "%\n")


nn = SentimentNetwork(reviews[:-1000], labels[:-1000], learning_rate=0.1)

# print("==================")
# nn.train()
# print("==================")
# nn.test(reviews[-1000:], labels[-1000:])

print("==================")
nn.train2()
print("==================")
nn.test2(reviews[-1000:], labels[-1000:])

import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing
Homework 1 - Programming Component: Trigram Language Models
Yassine Benajiba
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else 'UNK' for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """

    #Invalid input, return empty
    if len(sequence) < n or n < 1:
        return []

    startVar = 'START'
    stopVar = 'STOP'
    sequence_copy = sequence.copy()
    sequence_copy.append(stopVar)

    n_grams = list()

    if n == 1:
        n_grams.append(tuple(['START']))
    else:
        #create tuples using start key
        for i in range(-n, 0):
            curr_tuple = tuple(['START'],)
            j = 1
            index = i
            while j < n:
                index += 1
                j += 1
                if index < 0:
                    curr_tuple = curr_tuple + tuple(['START'],)
                else:
                    curr_tuple = curr_tuple + tuple([sequence_copy[index]])
                    if len(curr_tuple) == n:
                        n_grams.append(curr_tuple)





    #create rest of tuples
    for i in range(0, len(sequence_copy)):
        curr_tuple = tuple([sequence_copy[i]])
        tuple_is_valid = True
        for j in range((i+1), min(i+n, len(sequence_copy))):
            if j > len(sequence_copy):
                tuple_is_valid = False
                continue
            curr_tuple = curr_tuple + tuple([sequence_copy[j]])
        if tuple_is_valid and len(curr_tuple) == n:
            n_grams.append(curr_tuple)
    return n_grams

class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add('UNK')
        self.lexicon.add('START')
        self.lexicon.add('STOP')
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
        self.unigramcounts = {} # might want to use defaultdict or Counter instead
        self.bigramcounts = {} 
        self.trigramcounts = {}

        countSentences = 0
        for sentence in corpus:
            countSentences += 1
            sent_unigrams = get_ngrams(sentence, 1)
            sent_bigrams = get_ngrams(sentence, 2)
            sent_trigrams = get_ngrams(sentence, 3)

            for unigram in sent_unigrams:
                if unigram not in self.unigramcounts.keys():
                    self.unigramcounts[unigram] = 1
                else:
                    count = self.unigramcounts[unigram] + 1
                    self.unigramcounts[unigram] = count

            for bigram in sent_bigrams:
                if bigram not in self.bigramcounts.keys():
                    self.bigramcounts[bigram] = 1
                else:
                    count = self.bigramcounts[bigram] + 1
                    self.bigramcounts[bigram] = count

            for trigram in sent_trigrams:
                if trigram not in self.trigramcounts.keys():

                    self.trigramcounts[trigram] = 1
                else:
                    count = self.trigramcounts[trigram] + 1
                    self.trigramcounts[trigram] = count

        start_tuple = tuple(['START'],)
        start_tuple = start_tuple + tuple(['START'],)
        self.bigramcounts[start_tuple] = countSentences
        self.trigramtotals = sum(self.trigramcounts.values())
        self.bigramtotals = sum(self.bigramcounts.values())
        self.unigramtotals = sum(self.unigramcounts.values())

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """

        if trigram not in self.trigramcounts.keys():
            return 0.0

        count = self.trigramcounts[trigram]
        denom = self.bigramcounts[trigram[:2]] #TODO
        if count > denom:
            print("problem")
        return count / denom


    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        if bigram not in self.bigramcounts.keys():
            return 0.0

        count = self.bigramcounts[bigram]
        denom = self.unigramcounts[(bigram[0],)] #TODO
        if count > denom:
            print("problem")
        return count / denom

    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.

        if unigram not in self.unigramcounts.keys():
            return 0.0

        count = self.unigramcounts[unigram]
        return count / self.unigramtotals

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        #return result

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        trigram_value = lambda3 * self.raw_trigram_probability(trigram)
        bigram_value = lambda2 * self.raw_bigram_probability(trigram[:2])
        unigram_value = lambda1 * self.raw_unigram_probability(trigram[:1])
        value = trigram_value + bigram_value + unigram_value
        return value
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigrams = get_ngrams(sentence, 3)
        log_prob = 0
        for trigram in trigrams:
            smoothed_prop = self.smoothed_trigram_probability(trigram)
            curr_log_prob = 0
            if smoothed_prop != 0:
                curr_log_prob = math.log2(smoothed_prop)
            log_prob = log_prob + curr_log_prob
        return log_prob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        log_prob = 0
        denom = 0
        for sentence in corpus:
            n_grams = get_ngrams(sentence, 3)
            curr_log_prob = self.sentence_logprob(sentence)
            log_prob = log_prob + curr_log_prob
            denom += len(n_grams)
        value = log_prob / denom
        exp_value = 2**-value
        return exp_value


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1_high = TrigramModel(training_file1)
        model2_low = TrigramModel(training_file2)

        total = 0
        correct = 0       

        #testDir1=high
        for f in os.listdir(testdir1):
            pp = model1_high.perplexity(corpus_reader(os.path.join(testdir1, f), model1_high.lexicon))
            pp2 = model2_low.perplexity(corpus_reader(os.path.join(testdir1, f), model2_low.lexicon))
            if pp < pp2:
                correct += 1
            total += 1

        #testDir2=low
        for f in os.listdir(testdir2):
            pp_low = model2_low.perplexity(corpus_reader(os.path.join(testdir2, f), model2_low.lexicon))
            pp_high = model1_high.perplexity(corpus_reader(os.path.join(testdir2, f), model1_high.lexicon))
            if pp_low < pp_high:
                correct += 1
            total += 1
        return correct / total

if __name__ == "__main__":
    model = TrigramModel(sys.argv[1])
    #model = TrigramModel("/Users/Griffin/repos/4705_NLP/hw1/hw1_data/brown_train.txt")
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)


    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    #dev_corpus = corpus_reader("/Users/Griffin/repos/4705_NLP/hw1/hw1_data/brown_train.txt", model.lexicon)
    #pp = model.perplexity(dev_corpus)
    #print(pp)


    # Essay scoring experiment: 
    #acc = essay_scoring_experiment('/Users/Griffin/repos/4705_NLP/hw1/hw1_data/ets_toefl_data/train_high.txt', '/Users/Griffin/repos/4705_NLP/hw1/hw1_data/ets_toefl_data/train_low.txt', '/Users/Griffin/repos/4705_NLP/hw1/hw1_data/ets_toefl_data/test_high', '/Users/Griffin/repos/4705_NLP/hw1/hw1_data/ets_toefl_data/test_low')
    acc = essay_scoring_experiment('train_high.txt', 'train_low.txt', 'test_high', 'test_low')
    #locally getting 83% accuracy
    print(acc)


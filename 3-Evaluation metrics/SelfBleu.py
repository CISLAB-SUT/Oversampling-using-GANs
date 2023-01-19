import os
from multiprocessing import Pool

import nltk
from nltk.translate.bleu_score import SmoothingFunction

from utils.metrics.Metrics import Metrics


class SelfBleu(Metrics):
    def __init__(self, test_text='', gram=3):
        super().__init__()
        self.name = 'Self-Bleu'
        self.test_data = test_text
        self.gram = gram
        self.sample_size = 1000
        self.reference = None
        self.is_first = True

    def get_name(self):
        return self.name

    def get_score(self, is_fast=True, ignore=False):
        if ignore:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        if is_fast:
            return self.get_bleu_fast()
        return self.get_bleu_parallel()

    def get_reference(self):
        if self.reference is None:
            reference = list()
            with open(self.test_data) as real_data:
                for text in real_data:
                    text = nltk.word_tokenize(text)
                    reference.append(text)
            self.reference = reference
            return reference
        else:
            return self.reference

    def get_bleu(self):
        ngram = self.gram
        bleu = list()
        reference = self.get_reference()
        print("inja")
        print(reference)
        weight = tuple((1. / ngram for _ in range(ngram)))
        with open(self.test_data) as test_data:
            for hypothesis in test_data:
                print(hypothesis)
                hypothesis = nltk.word_tokenize(hypothesis)
                bleu.append(nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight))
                print(bleu)

        print(sum(bleu))
        print(len(bleu))
        result = sum(bleu) / len(bleu)
        print(result)
        return sum(bleu) / len(bleu)

    def get_SelfBleu(self):
        ngram = self.gram
        bleu = list()
        reference = self.get_reference()
        # print("inja")
        # print(reference)
        weight = tuple((1. / ngram for _ in range(ngram)))
        sentence_num = len(reference)
        for index in range(sentence_num):
            hypothesis = reference[index]
            other = reference[:index] + reference[index + 1:]
            bleu.append(nltk.translate.bleu_score.sentence_bleu(other, hypothesis, weight,
                                                                smoothing_function=SmoothingFunction().method1))
            # print(bleu)

        print(sum(bleu))
        print(len(bleu))
        result = sum(bleu) / len(bleu)
        print(result)
        return sum(bleu) / len(bleu)


    def calc_bleu(self, reference, hypothesis, weight):
        return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                       smoothing_function=SmoothingFunction().method1)

    def get_bleu_fast(self):
        reference = self.get_reference()
        # random.shuffle(reference)
        reference = reference[0:self.sample_size]
        return self.get_bleu_parallel(reference=reference)

    def get_bleu_parallel(self, reference=None):
        ngram = self.gram
        if reference is None:
            reference = self.get_reference()
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(os.cpu_count())
        result = list()
        sentence_num = len(reference)
        for index in range(sentence_num):
            hypothesis = reference[index]
            other = reference[:index] + reference[index+1:]
            result.append(pool.apply_async(self.calc_bleu, args=(other, hypothesis, weight)))

        score = 0.0
        cnt = 0
        for i in result:
            score += i.get()
            cnt += 1
        pool.close()
        pool.join()
        return score / cnt

# 4 is number on n-grams
bleu=SelfBleu('sentiment140/sentigan_seqgan_1000.txt',4)
# bleu.get_bleu()
bleu.get_SelfBleu()
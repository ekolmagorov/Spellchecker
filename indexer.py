# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
import os


def filter_function(source_string):
    return source_string.lower()


def create_dataset(strings):

    data = pd.DataFrame([line.rstrip().split('\t') for line in strings], columns=['source', 'corrected'])
    corrected = data.dropna()
    wrong_rigth = list(zip(corrected['source'],corrected['corrected']))
    wrong_rigth = [(filter_function(wrong), filter_function(rigth)) for wrong, rigth in wrong_rigth]
    rigth = list(data[data.corrected.isnull()]['source'])
    rigth = rigth + list(corrected['corrected'])
    rigth = list(map(filter_function, rigth))

    return rigth, wrong_rigth

#-------------------------------------------------------------------------------
def damerau_levenshtein_distance(s1, s2):

    lenstr1 = len(s1)
    lenstr2 = len(s2)
    d = np.zeros((lenstr1 + 1, lenstr2 + 1), dtype=np.uint8)

    for i in range(0, lenstr1 + 1):

        d[i, 0] = i

    for j in range(0, lenstr2 + 1):

        d[0, j] = j


    for i in range(1, lenstr1 + 1):

        for j in range(1, lenstr2 + 1):

            if s1[i-1] == s2[j-1]:
                cost = 0

            else:
                cost = 1

            d[i, j] = min( d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + cost)

            if i > 1  and j > 1 and s1[i-1]==s2[j-2] and s1[i-2] == s2[j-1]:
                d[i, j] = min (d[i, j], d[i-2, j-2] + cost) # transposition

    return d
#-------------------------------------------------------------------------------

def getErrorType(s1, s2, dist_matrix = None):

    if dist_matrix is None:
        dist_matrix = damerau_levenshtein_distance(s1, s2)

    i, j = dist_matrix.shape

    i -= 1
    j -= 1

    ops = list()

    while i != -1 and j != -1:
        if i > 1 and j > 1 and s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1]:
            if dist_matrix[i-2, j-2] < dist_matrix[i, j]:
                ops.insert(0, ('transpose', i - 1, i - 2))
                i -= 2
                j -= 2
                continue

        index = np.argmin([dist_matrix[i-1, j-1], dist_matrix[i, j-1], dist_matrix[i-1, j]])

        if index == 0:
            if dist_matrix[i, j] > dist_matrix[i-1, j-1]:
                ops.insert(0, ('replace', i - 1, j - 1))
            i -= 1
            j -= 1
        elif index == 1:
            ops.insert(0, ('insert', i - 1, j - 1))
            j -= 1
        elif index == 2:
            ops.insert(0, ('delete', i - 1, i - 1))
            i -= 1

    return ops

#-------------------------------------------------------------------------------
def create_errors_model(wrong_rigth):

    error_probs = dict()

    i = 1
    for error_rigth in wrong_rigth:
        error_words = error_rigth[0].split()
        correct_words = error_rigth[1].split()
        if len(error_words) != len(correct_words):
            continue


        #if (i % 1000 == 0):
        #    print(i)
        #i += 1

        for  error_word, rigth_word in zip(error_words, correct_words):


            error_word = ''.join(c for c in error_word if c not in trash_chars)
            rigth_word = ''.join(c for c in rigth_word if c not in trash_chars)

            if (error_word == rigth_word or not len(error_word) or not len(rigth_word)):
                continue

            modifications = getErrorType(error_word, rigth_word)
            for modification in modifications:
                if modification[0] == 'transpose':

                    traspose_position_left = modification[2]
                    traspose_position_rigth = modification[1]

                    chunk_error = error_word[traspose_position_left : traspose_position_rigth + 1]
                    chunk_rigth = rigth_word[traspose_position_left : traspose_position_rigth + 1]

                    error = (chunk_error, chunk_rigth)

                elif modification[0] == 'replace':

                    error_letter = error_word[modification[1]]
                    rigth_letter = rigth_word[modification[2]]

                    error = (error_letter, rigth_letter)


                elif modification[0] == 'delete':

                    error_letter = error_word[modification[1]]
                    error = (error_letter, '')

                else:
                    rigth_letter = rigth_word[modification[2]]
                    error = ('', rigth_letter)

                if error in error_probs:
                    error_probs[error] += 1

                else:
                    error_probs[error] = 1

    total_num_errors = sum([error_probs[key] for key in error_probs.keys()])
    error_probs = {key:-np.log(error_probs[key] / (total_num_errors + 1e-6)) for key in error_probs.keys()}

    return error_probs
#-------------------------------------------------------------------------------

def filter_from_trash(word, trash_chars):
    return ''.join(char for char in word if char not in trash_chars)

def bigram_stats(corpus, trash_chars):

    d = dict()

    for sentence in corpus:
        words = sentence.split()

        words = [filter_from_trash(word, trash_chars) for word in words
                 if filter_from_trash(word, trash_chars) != '']

        for i in range(len(words) - 1):
            bigram = tuple(words[i:i+2])
            if bigram in d:
                d[bigram] += 1
            else:
                d[bigram] = 1

    total_bigrams = sum([d[key] for key in d.keys()])
    d = {key: -np.log(d[key]/(total_bigrams + 1e-6)) for key in d.keys()}

    return d, total_bigrams


def unigram_stats(corpus, trash_chars):
    d = dict()

    for sentence in corpus:
        words = sentence.split()

        words = [filter_from_trash(word, trash_chars) for word in words
                 if filter_from_trash(word, trash_chars) != '']

        for word in words:

            if word in d:
                d[word] += 1
            else:
                d[word] = 1


    total_words = sum([d[key] for key in d.keys()])
    d = {key: -np.log(d[key]/(total_words + 1e-6)) for key in d.keys()}

    return d, total_words




if __name__ == '__main__':

    filename = 'queries_all.txt'
    trash_chars = set(u"?!@#$%^&*()_+/\><[]{}.,:;\'\"-\ 1234567890")

    chars_eng = set(chr(i) for i in range(ord('a'),ord('z')))
    chars_rus = set(u'абвгдеёжзийклмнопрстуфхцчшщъыьэюя')

    rus_keyboard = u'йцукенгшщзхъфывапролджэячсмитьбю.,ё'
    eng_keyboard = u'qwertyuiop[]asdfghjkl;\'zxcvbnm,./?`'

    ru_en = {rus_char: eng_char for rus_char, eng_char in zip(rus_keyboard, eng_keyboard)}
    en_ru = {eng_char: rus_char for eng_char, rus_char in zip(eng_keyboard, rus_keyboard)}

    strings = []
    with open(filename) as queries:
        for line in queries:
            strings.append(line.decode('utf-8'))

    rigth, wrong_rigth = create_dataset(strings)

    #print("[info] Create error_model")
    probs = create_errors_model(wrong_rigth)
    if 'errors_model.pickle' not in os.listdir('.'):
        with open('errors_model.pickle', 'wb') as f:
            pickle.dump(probs, f, protocol=2)

    del probs

    #print("[info] Create bigrams statistics")
    bstats, total_bigrams = bigram_stats(rigth, trash_chars)
    if 'bstats.pickle' not in os.listdir('.'):
        with open('bstats.pickle', 'wb') as f:
            pickle.dump((bstats, total_bigrams), f, protocol=2)

    del bstats, total_bigrams

    #print("[info] Create unigrams statistics")
    ustats, total_words = unigram_stats(rigth, trash_chars)

    if 'ustats.pickle' not in os.listdir('.'):
        with open('ustats.pickle', 'wb') as f:
            pickle.dump((ustats, total_words), f, protocol=2)

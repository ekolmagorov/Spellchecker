# -*- coding: utf-8 -*-

from heapq import heappush, heappop
import indexer
import pickle
import numpy as np

#-------------------------------------------------------------------------------
class Bor(object):

    def __init__(self, letter = None, parent = None):
        self.letter = letter
        if parent is None:
            self.parent = self
        else:
            self.parent = parent

        self.nodes = {}
        self.freq = None
        self.word = None

    def __contains__(self, word):

        current_node = self
        for i in range(len(word)):

            if word[i] in current_node.nodes:
                current_node = current_node.nodes[word[i]]
            else:
                return False

        return True

    def getWordFreq(self, word):
        if word not in self:
            return None

        current_node = self
        for i in range(len(word)):
            current_node = current_node.nodes[word[i]]

        return current_node.freq


    def addWord(self, word, word_freq):

        if len(word) == 0:
            return

        current_node = self
        tmp_word = word[:]
        while (tmp_word != ''):

            if current_node.letter is None:
                if tmp_word[0] not in current_node.nodes:
                    self.nodes[tmp_word[0]] = Bor(tmp_word[0], current_node)

            else:
                tmp_word = tmp_word[1:]
                if len(tmp_word) == 0:
                    current_node.freq  = word_freq
                    current_node.word = word
                    break

                if tmp_word[0] not in current_node.nodes:
                    current_node.nodes[tmp_word[0]]= Bor(tmp_word[0], current_node)

            current_node = current_node.nodes[tmp_word[0]]


    def printAllWords(self, result=''):

        if self.letter is not None:
            result += self.letter

        if self.freq is not None:
            print(result, self.freq)

        for letter in sorted(self.nodes.keys()):

            self.nodes[letter].printAllWords(result)

#-------------------------------------------------------------------------------

def search(bor, word, error_model,  alpha = 1.5):

    def substitution(bor, word, score, num_errors):

        if ((bor, word) in visited) or score > max_score or num_errors > max_errors:
            return

        visited.add((bor, word))

        if len(word) == 0:
            if bor.word != None:
                scores_substitute.append((bor.word, alpha * bor.freq + score))
            return

        for key in bor.nodes.keys():
            #print(key, word[0])
            if key == word[0]:# and not delet and not ins and not transposed:
                substitution(bor.nodes[key], word[1:], score, num_errors)

            elif (word[0], key) in error_model:
                error_score = error_model[(word[0],key)]
                substitution(bor.nodes[key], word[1:], score + error_score, num_errors + 1)


    def deletion(bor, word, score, num_errors):

        if ((bor, word) in visited) or score > max_score or num_errors > max_errors:
            return

        visited.add((bor, word))

        if len(word) == 0 and bor.word != None:
            scores_delete.append((bor.word, alpha * bor.freq + score))
            return

        if len(word) :
            if (word[0] in bor.nodes):
                deletion(bor.nodes[word[0]], word[1:], num_errors, score)

            error = (word[0], '')
            if error in error_model:
                diff_score = error_model[error]
                deletion(bor, word[1:], score + diff_score, num_errors + 1)

    def insertion(bor, word, score, num_errors, lang_chars):

        if ((bor, word) in visited) or score > max_score or num_errors > max_errors - 2:
            return

        if len(word) == 0 and bor.word != None:
            scores_insert.append((bor.word, alpha * bor.freq + score))

        visited.add((bor, word))

        if (len(word) and word[0] in bor.nodes):
            insertion(bor.nodes[word[0]], word[1:], score, num_errors, lang_chars)


        for char in lang_chars:
            error = ('', char)
            if error in error_model:
                error_score = error_model[error]
                insertion(bor, char + word, score + error_score, num_errors + 1, lang_chars)




    scores_substitute = [(word, -np.log(1 /(3 * total_words + 1e-6)) + 1e4)]
    scores_delete = [(word, -np.log(1 /(3 * total_words + 1e-6)) + 1e4)]
    scores_insert = [(word, -np.log(1 /(3 * total_words + 1e-6)) + 1e4)]


    max_errors = 3
    max_score = 17
    if (set(word).intersection(chars_rus)):
        lang_chars = chars_rus
    else:
        lang_chars = chars_eng

    if len(word) <= 3:
        return set([(word, -np.log(1/(total_words + 1e-6)))])

    #print(insert)
    visited = set()
    substitution(bor, word, 0, 0)
    visited = set()
    deletion(bor, word, 0 , 0)
    visited = set()
    insertion(bor, word, 0, 0, lang_chars)


    scores = sorted(scores_substitute, key=lambda x:x[1])[:4] +\
    sorted(scores_delete, key=lambda x: x[1])[:4] + sorted(scores_insert, key=lambda x:x[1])[:4]

    return set(scores)

#-------------------------------------------------------------------------------



def changeLang(query):

    words = query.split()
    another_lang = []

    for word in words:
        if set(word).intersection(chars_rus):
            lang_chars = 'ru'
        else:
            lang_chars = 'en'

        other_lang = []

        if lang_chars == 'en':
            for char in word:
                if (char in en_ru):
                    other_lang.append(en_ru[char])
                else:
                    other_lang.append(char)
        else:
            for char in word:
                if char in ru_en:
                    other_lang.append(ru_en[char])
                else:
                    other_lang.append(char)

        another_lang.append(''.join(other_lang))

    return ' '.join(another_lang)

#-------------------------------------------------------------------------------
#Generator
def generate(query, bor, error_model, bstats, total_bigrams, betta = 2):

    #print(len(query), query)
    if not len(query):
        return (0 ,'')

    h = []
    corrected_words = []

    for word in query.split():
        corrected_scores  = search(bor, word, error_model)
        corrected_words.append(corrected_scores)



    for word in corrected_words[0]:
        heappush(h, (word[1], word[0]))

    for words in corrected_words[1:]:
        new_h = []

        for i in range(len(h)):
            score_query = heappop(h)
            query = score_query[1]
            query_score = score_query[0]
            last_word = query.split()[-1]

            for word in words:
                if (last_word, word[0]) in bstats:
                    score = betta * bstats[(last_word, word[0])]

                else:
                    score = -np.log(1/ (2 * total_bigrams + 1e-6)) * betta

                heappush(new_h, (query_score + score + word[1], query +' ' + word[0]))

        h = [heappop(new_h) for i in range(min(len(new_h),10))]

    return [heappop(h) for i in range(min(len(h), 10))]

#-------------------------------------------------------------------------------
#Iterations
def correct(query, bor, error_model, bstats, total_bigrams, max_iteration = 5):

    old_generated = (0,'')
    new_generated = (0, query)

    words = query.split()
    i = 0
    while (new_generated != old_generated):
        if (i == max_iteration):
            break

        old_generated = new_generated
        new_generated = generate(old_generated[1], bor, error_model, bstats, total_bigrams)[0]
        i += 1


    return new_generated

#-------------------------------------------------------------------------------
#Move trash chars from query
def find_trash(query):

    result = ''
    without_trash = []
    for char in query:
        if (char in trash_chars or char == ' '):
            if len(result):
                without_trash.append(result)
                result = ''

            without_trash.append(char)

        else:
            result += char

    without_trash.append(result)
    return  without_trash

#-------------------------------------------------------------------------------
#final function
def spellcheck(query, bor, error_model, bstats, total_bigrams):


    source = unicode(query.lower())
    converted = unicode(changeLang(source))

    source_words = find_trash(source)
    converted_words = find_trash(converted)

    indexes_converted = []

    source_query = []
    source_indexes = []
    for idx, source_word in enumerate(source_words):
        if (source_word not in trash_chars and len(source_word)):
            source_query.append(source_word)
            source_indexes.append(idx)

    converted_query = []
    converted_indexes = []
    for idx, converted_word in enumerate(converted_words):
        if (converted_word not in trash_chars and len(converted_word)):
            converted_query.append(converted_word)
            converted_indexes.append(idx)

    source_score = correct(' '.join(source_query), bor, error_model, bstats, total_bigrams, 1)
    converted_score = correct(' '.join(converted_query), bor, error_model, bstats, total_bigrams, 1)
    #print(source_score, converted_score)

    if (source_score[0] <= converted_score[0]):
        result = correct(source_score[1], bor, error_model, bstats, total_bigrams)
        insert = source_words
        indexes = source_indexes
    else:
        result = correct(converted_score[1], bor, error_model, bstats, total_bigrams)
        insert = converted_words
        indexes = converted_indexes

    for idx, word in zip(indexes, result[1].split()):
        insert[idx] = word

    return ''.join(insert)


if __name__ == '__main__':

    trash_chars = set("?!@#$%^&*()_+/\><[]{}.,:;\'\"-\ 1234567890\n\t\r")

    chars_eng = set(chr(i) for i in range(ord('a'),ord('z')))
    chars_rus = set(u'абвгдеёжзийклмнопрстуфхцчшщъыьэюя')

    rus_keyboard = u'йцукенгшщзхъфывапролджэячсмитьбю.,ё'
    eng_keyboard = u'qwertyuiop[]asdfghjkl;\'zxcvbnm,./?`'

    ru_en = {rus_char: eng_char for rus_char, eng_char in zip(rus_keyboard, eng_keyboard)}
    en_ru = {eng_char: rus_char for eng_char, rus_char in zip(eng_keyboard, rus_keyboard)}

    with open('errors_model.pickle', 'rb') as f:
        error_model = pickle.load(f)
        min_prob = min([error_model[key] for key in error_model])

    with open('bstats.pickle', 'rb') as f:
        bstats, total_bigrams = pickle.load(f)

    with open('ustats.pickle', 'rb') as f:
        ustats, total_words = pickle.load(f)

    #print("models loaded")
    bor = Bor()
    for key in ustats.keys():
        bor.addWord(key, ustats[key])


    query = raw_input()
    while (query):

        res = spellcheck( query.decode('utf-8'), bor, error_model, bstats, total_bigrams)
        print(res)
        query = raw_input()
    #with open('queries_all.txt', 'r') as f:
    #    for query in f:
    #        print query
    #        print spellcheck(query.decode('utf-8'), bor, error_model, bstats, total_bigrams)

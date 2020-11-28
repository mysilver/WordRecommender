import heapq
import math
import unicodedata
from collections import Counter

import time
from flask import Flask, request, jsonify
from flask_restplus import Resource, Api
from flask_restplus import fields
from nltk import ngrams as ngram_f, TweetTokenizer, ngrams, OrderedDict
from nltk.corpus import stopwords as _stopwords, words
from nltk.stem import WordNetLemmatizer

from utils import wsd
from utils.entities import Parameter, Task
from utils.service import word2vec_client
from utils.service.language_tool import LanguageChecker
from utils.service.word2vec_client import vocabulary
from utils.service.word_aligner import WordAligner

app = Flask(__name__)
api = Api(app,
          default="WordCloud",  # Default namespace
          title="Word-Cloud Generator",  # Documentation Title
          description="A dynamic visual representation of grouped words to improve lexical diversity")  # Documentation Description

request_model = api.model('WordCloud', {
    'expression': fields.String,
    'parameters': fields.Raw,
    'paraphrases': fields.Raw,
    'cloud_size': fields.Integer,
})


class Utils:
    def __init__(self, dictionary, word2vec_client=None, wsd=None, database=None) -> None:
        self.database = database
        self.word_aligner = WordAligner()
        self.word2vec_client = word2vec_client
        self.wsd = wsd
        self.dictionary = dictionary
        self.wl = WordNetLemmatizer()
        self.stopwords = set(_stopwords.words('english')).union(['.', '?', '!', ';', ':'])
        self.p_stopwords = {'to_the', 'there_is', 'on_to', 'for_one', 'for_me', 'of_all', 'like_to', 'for_days',
                            'place_to',
                            'related_to'}
        self.spellChecker = LanguageChecker()

    def align(self, sentence_1, sentence_2):
        return self.word_aligner.align(sentence_1, sentence_2)

    @staticmethod
    def get_n_longest_values_gh(dictionary, n):
        return heapq.nlargest(n, ((value, key) for key, value in dictionary.items()))

    @staticmethod
    def is_valid_word(word):
        if len(word) < 2 or (len(word) > 25 and ' ' not in word):
            return False

        if word not in words.words() and '_' not in word:
            return False
        return True

    def tokenize_2(self, text, ngrams_sizes=(3, 2), remove_stopwords=True):
        tknzr = TweetTokenizer()
        text = text.lower()
        if ngrams_sizes:
            for i in ngrams_sizes:
                # join ngrams with '_'
                tokens = tknzr.tokenize(text)
                ngs = ngrams(tokens, i)
                for ng in ngs:
                    phrs = "_".join(ng)
                    if phrs in self.dictionary:
                        text = text.replace(" ".join(ng), phrs)

        tokens = tknzr.tokenize(text)
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        return tokens

    def tokenize(self, text: str, remove_unseen=False):

        tokens = text.split()
        if not remove_unseen:
            return tokens

        return [i for i in tokens if i in self.dictionary]

    def pre_process_word(self, word, heuristic=False):

        word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('ascii')

        t = word.lower()
        if t in self.dictionary:
            word = t

        if heuristic:
            if '_' in word:
                vpart = word[:word.index('_')]
                rest = word.replace(vpart + '_', '')
                vpart = self.wl.lemmatize(vpart, 'v')
                return vpart + '_' + rest
            return self.wl.lemmatize(t, 'v')

        return word

    def extract_ngrams(self, tokens):
        ret = []
        for i in range(2, 4):
            for ngram in ngram_f(tokens, i):
                phrase = '_'.join(ngram)
                if phrase not in self.p_stopwords:
                    if phrase in self.dictionary:
                        ret.append(phrase)
        return ret

    def resultant_similarity(self, related_words, context_words):

        ret = []
        uwords = set()
        for w, score in related_words:
            if w not in uwords:
                uwords.add(w)
                if w in self.dictionary and Utils.is_valid_word(w):
                    # ret.append((score * (self.word2vec_client.n_similarity([w], context_words)), w))
                    ret.append((self.word2vec_client.n_similarity([w], context_words), w))

        return ret

    def disambiguate(self, word, sentence):
        """
        Word Sense Disambiguation 
        """
        top_senses = [word]
        # top_senses, score = sts(word[0], pos_to_a_letter(word[1]), sentence)
        # print(score, top_senses)
        # l = lesk(word[0], sentence)
        # if l is not None:
        #     top_senses.extend(l)
        # print('sts WSD- ', word, ':', top_senses)
        ss, score2 = self.wsd.closest_resultant(word, sentence.split(), self.tokenize,
                                                self.word2vec_client.wm_distance)
        # print(score2, ss)
        top_senses.extend(ss)
        # top_senses = set(top_senses)
        # print(word, ":", top_senses)
        return list(filter(lambda x: x in self.dictionary and x not in self.stopwords, top_senses))

    @staticmethod
    def merge_candidatesets(re_ranked_candidates):
        to_remove = set()
        for k1 in re_ranked_candidates.keys():
            for k2 in re_ranked_candidates.keys():
                if k1 != k2:
                    if k1.find('_' + k2) > -1 or k1.find(k2 + '_') > -1:
                        to_remove.add(k2)
                        re_ranked_candidates[k1].extend(re_ranked_candidates[k2])

        for k in to_remove:
            re_ranked_candidates.pop(k)
        return re_ranked_candidates

    @staticmethod
    def merge_keywordsets(keysets):
        to_remove = []
        for k1 in keysets:
            for k2 in keysets:
                if k1 != k2:
                    for k1i in k1:
                        if k1i.find('_' + k2[0]) > -1 or k1i.find(k2[0] + '_') > -1:
                            to_remove.append(k2)
                            if k2 not in k1:
                                k1.extend(k2)

        for k in to_remove:
            keysets.remove(k)
        return keysets


class Database:
    def __init__(self) -> None:
        self.expr_paraphrase_map = {}
        self.expr_wordfreq_map = {}
        self.expr_cloud_word_map = {}

    def add(self, expression, paraphrases, word_freq, cloud_word_statistics):
        self.expr_paraphrase_map[expression] = paraphrases
        self.expr_wordfreq_map[expression] = word_freq
        self.expr_cloud_word_map[expression] = cloud_word_statistics

    def word_frequency(self, expression, word):
        return self.expr_wordfreq_map.get(expression, {}).get(word, 0)

    def cloud_word_statistics(self, expression, word):
        return self.expr_cloud_word_map.get(expression, {}).get(word, (0, 0))

    def num_of_paraphrases(self, expression):
        return len(self.expr_paraphrase_map.get(expression, []))


class KeywordSetExtractor:
    def __init__(self, utils: Utils, database: Database) -> None:
        self.database = database
        self.utils = utils

    def extract(self, task):
        tokens = self.utils.tokenize(task.expression)
        tokens = [self.utils.pre_process_word(t) for t in tokens]

        for p in task.parameters:
            if p.editable:
                tokens = [p.value if word == p.name else word for word in tokens]
            else:
                tokens = [word for word in tokens if word != p.name]

        tokens.extend(self.utils.extract_ngrams(tokens))

        alignments = {}
        for paraph in list(self.database.expr_paraphrase_map[task.expression].keys()):
            for a in self.utils.align(task.to_string().lower(), paraph.lower()):
                if a[0] in alignments:
                    alignments[a[0]].add(a[1])
                else:
                    alignments[a[0]] = {a[1]}

                    # do alignment

        tokens = list(filter(lambda x: x in self.utils.dictionary and x not in self.utils.stopwords, tokens))

        keyword_set = []
        for t in tokens:
            if t in alignments:
                ss = [t]
                ss.extend(list(alignments[t]))
            else:
                ss = self.utils.disambiguate(t, task.to_string())

            if not ss:
                ss = [t]
            elif t not in ss:
                ss.append(t)

            keyword_set.append(ss)

        # common_words = self.top_words_not_included(temp, task.to_string().lower(), dictionary_per_expression)
        self.utils.merge_keywordsets(keyword_set)

        # remove duplication from keyword sets
        ret = []
        for ks in keyword_set:
            ret.append(list(OrderedDict.fromkeys(ks)))
        print("Keyword sets:", ret)
        return ret


class CandidateSetGenerator:
    def __init__(self, utils: Utils) -> None:
        self.utils = utils

    def generate(self, task, keywords_set):
        sentence = task.to_string().lower()
        ret = {}

        context_words = [i for i in self.utils.tokenize_2(task.expression) if i not in self.utils.stopwords] #.replace('$', '')
        for keyset in keywords_set:

            # start = time.time()
            candidate_words = self.utils.word2vec_client.most_similar(keyset, 50)
            # print("similar Set:", time.time() - start)

            # start = time.time()
            candidate_words = [(self.utils.spellChecker.singleWordCorrection(w[0]), w[1]) for w in candidate_words]
            # print("spell Set:", time.time() - start)

            # start = time.time()
            # candidate_words = [(w[0], w[1]) for w in candidate_words if w[0] not in sentence]
            # print("not in sentence Set:", time.time() - start)

            # start = time.time()
            ret[keyset[0]] = self.utils.resultant_similarity(candidate_words, context_words)
            # print("context similarity Set:", time.time() - start)

        ret = self.utils.merge_candidatesets(ret)
        return ret


class CandidateSetRanker:
    @staticmethod
    def similarity_probability(candidate_set: (float, str)):
        sum_vals = sum([score for score, _ in candidate_set])
        return dict([(word, score / sum_vals) for score, word in candidate_set])

    @staticmethod
    def merge_probabilities(similarity_probability, diversity_probability, relevance_probability, alpha, betta, theta,
                            size):
        assert alpha + betta + theta == 1

        queue = {}
        for k in similarity_probability:
            queue[k] = alpha * similarity_probability[k] + betta * diversity_probability[k] + theta * \
                                                                                              relevance_probability[k]

        return Utils.get_n_longest_values_gh(queue, size)

    @staticmethod
    def diversity_probability(expression, candidate_set, database: Database):
        candidates_idfs = []
        sum_vals = 0
        for _, w in candidate_set:
            df = database.word_frequency(expression, w)
            np = database.num_of_paraphrases(expression)
            idf = math.log(1 + (np - df + 0.5) / (df + 0.5))
            candidates_idfs.append((w, idf))
            sum_vals += idf

        candidates_idfs = dict([(word, score / sum_vals) for word, score in candidates_idfs])
        return candidates_idfs

    @staticmethod
    def relevance_probability(expression, candidate_set, database: Database, lamda: float, gamma: float,
                              epsilon: float):
        candidates_relevances = []
        sum_vals = 0
        for _, w in candidate_set:
            shown, used = database.cloud_word_statistics(expression, w)
            relevance = (lamda * used + gamma * (shown - used) + epsilon) / (shown + epsilon)
            candidates_relevances.append((w, relevance))
            sum_vals += relevance

        candidates_relevances = dict([(word, score / sum_vals) for word, score in candidates_relevances])
        return candidates_relevances

    @staticmethod
    def rank(task: Task, candidate_sets, database: Database, word_cloud_size: int, lamda: float, gamma: float,
             epsilon: float, alpha: float, betta: float,
             theta: float):

        expression = task.expression
        size_per_candidateset = int(word_cloud_size / len(candidate_sets)) + 1
        ret = {}
        for c in candidate_sets:
            candset = candidate_sets[c]
            similarity_probability = CandidateSetRanker.similarity_probability(candset)
            diversity_probability = CandidateSetRanker.diversity_probability(expression, candset, database)
            relevance_probability = CandidateSetRanker.relevance_probability(expression, candset, database, lamda,
                                                                             gamma, epsilon)

            probabilities = CandidateSetRanker.merge_probabilities(similarity_probability, diversity_probability,
                                                                   relevance_probability, alpha, betta, theta,
                                                                   size_per_candidateset)
            ret[c] = probabilities
        return ret


class Test:
    def testKeywordSetExtractor(self):
        utils = Utils(vocabulary(), word2vec_client, wsd)
        task = Task(0, "Seek for a restaurant near $location",
                    parameters=[Parameter("$location", "Sydney", editable=True)])

        print(KeywordSetExtractor(utils).extract(task))

    def testCandidateSetGenerator(self):
        utils = Utils(vocabulary(), word2vec_client, wsd)
        task = Task(0, "Seek for a restaurant near $location",
                    parameters=[Parameter("$location", "Sydney", editable=True)])

        keysets = KeywordSetExtractor(utils).extract(task)
        print(CandidateSetGenerator(utils).generate(task, keysets))

    def testCandidateSetRanker(self):
        utils = Utils(vocabulary(), word2vec_client, wsd)
        task = Task(0, "Seek for a restaurant near $location",
                    parameters=[Parameter("$location", "Sydney", editable=True)])

        keysets = KeywordSetExtractor(utils).extract(task)
        candsets = CandidateSetGenerator(utils).generate(task, keysets)
        database = Database()
        ranks = CandidateSetRanker.rank(task, candsets, database, 12, 0.8, 0.5, 0.1, 0.33, 0.33, 0.34)
        print(ranks)


class WordCloudGenerator:
    def __init__(self, utils: Utils):
        self.utils = utils

    def generate(self, task: Task, database: Database, word_cloud_size: int):
        start = time.time()
        keysets = KeywordSetExtractor(self.utils, database).extract(task)
        print("Keyword Set:", time.time() - start)

        start = time.time()
        candsets = CandidateSetGenerator(self.utils).generate(task, keysets)
        print("Candidate Set:", time.time() - start)
        start = time.time()

        ranks = CandidateSetRanker.rank(task, candsets, database, word_cloud_size, 0.8, 0.5, 0.1, 0.33, 0.33, 0.34)
        print("Ranking:", time.time() - start)
        return ranks


@api.route('/wordcloud')
class WordCloud(Resource):
    def __init__(self, api=None, *args, **kwargs):
        super().__init__(api, *args, **kwargs)
        self.utils = Utils(vocabulary(), word2vec_client, wsd)
        self.wcg = WordCloudGenerator(self.utils)

    @api.response(200, 'Successful')
    @api.doc(description="Word-Cloud")
    @api.expect(request_model, validate=True)
    def post(self):
        """
        Input example: 
        {
          "paraphrases": {
            "Any suggestions for a place to eat desserts in Sydney" : ["suggest","eatery"]
          },
          "parameters": {"$location": "Sydney"},
          "expression": "Any suggestions for a place to eat desserts in $location"
        }
        :return: 
        """
        # get books as JSON string
        args = request.json

        # retrieve the query parameters
        expr = args.get('expression', "").lower()
        word_cloud_size = args.get('cloud_size', 12)
        params = args.get('parameters', [])
        parameters = [Parameter(name=key, value=val) for key, val in params.items()]
        paraphrases = args.get('paraphrases', {})

        task = Task(0, expression=expr, goal="", parameters=parameters)

        if not expr:
            return {"message": "Expression is missing"}, 400

        word_freq = Counter()
        shown_freq = Counter()
        used_freq = Counter()

        for p in paraphrases:
            word_cloud_words = paraphrases[p]
            for t in word_cloud_words:
                shown_freq.update([t])

            for t in self.utils.tokenize_2(p.lower()):
                word_freq.update([t])
                if t in word_cloud_words:
                    used_freq.update([t])

        cloud_word_statistics = {}
        for t in shown_freq:
            shown = shown_freq[t]
            used = used_freq[t]
            cloud_word_statistics[t] = (shown, used)

        database = Database()
        database.add(task.expression, paraphrases, word_freq, cloud_word_statistics)
        rlt = self.wcg.generate(task, database, word_cloud_size)

        ret = {}
        for k in rlt:
            ret[k] = [{"word": i[1], "score": i[0]} for i in rlt[k]]

        return jsonify(ret)


if __name__ == "__main__":
    app.run(port="9090")

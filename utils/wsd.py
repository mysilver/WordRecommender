from queue import PriorityQueue
from nltk.corpus import wordnet as wn


def closest_resultant(word, sequence, to_keywords, wm_distance):
    q = PriorityQueue()
    for ss in wn.synsets(word.replace('_', ' ')):
        t = ss.lemma_names()
        keywords = to_keywords(ss.definition(), True)
        # score = wm_distance(sequence, keywords)
        score = wm_distance(sequence, keywords)
        for ex in ss.examples():
            # score += wm_distance(sequence, to_keywords(ex, True))
            score += wm_distance(sequence, to_keywords(ex, True))

        score /= len(ss.examples()) + 1
        # q.put((-score, t))
        q.put((score, t))

    if q.empty():
        return [], 0
    # return list(filter(lambda x: x in vocab and x not in set(stopwords.words('english')), q.get()[1]))
    ret = q.get()
    return ret[1], ret[0]

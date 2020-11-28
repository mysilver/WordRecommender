from functools import lru_cache
import requests


class Client(object):
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.headers = {
            'content-type': 'application/json'
        }

    def _get_url(self, path='/aligner'):
        return "http://{0}:{1}{2}".format(self.host, self.port, path)

    def align(self, sentence1, sentence2):
        url = self._get_url()
        response = requests.post(url, headers=self.headers, json={"sentence_1": sentence1, "sentence_2": sentence2})

        if not response.ok:
            print("Word Aligner error ", sentence1, sentence2)
            return []

        content = response.json()
        return content


class WordAligner:
    def __init__(self, host='localhost', port=5100):
        self.client = Client(host, port)

    @lru_cache(maxsize=500)
    def align(self, sentence1, sentence2):
        return self.client.align(sentence1, sentence2)


"""Dependencies required to use of this file can be installed as following:
   pip install gensim clint requests pandas nltk
   python -m nltk.downloader punkt """

import os
import csv
from os.path import exists
from typing import List, Tuple
from collections import namedtuple
from operator import itemgetter

from gensim.models import KeyedVectors
from numpy import mean
from pandas import read_csv

SenseBase = namedtuple('Sense', 'word keyword cluster')


class Sense(SenseBase):  # this is needed as list is an unhashable type
    def get_hash(self):
        return hash(self.word + self.keyword + "".join(self.cluster))

    def __hash__(self):
        return self.get_hash()

    def __eq__(self, other):
        return self.get_hash() == other.get_hash()


def ensure_word_embeddings(language: str):
    """ Ensures that the word vectors exist or raise Exception. """

    dir_path = os.path.join("models", "fasttext_models", language)
    filename = "cc.{}.300.vec.gz".format(language)
    wv_fpath = os.path.join(dir_path, filename)
    wv_pkl_fpath = wv_fpath + ".pkl"

    if not exists(wv_fpath):
        raise Exception('No model for {} language: {}'.format(language, wv_fpath))

    return wv_fpath, wv_pkl_fpath


MOST_SIGNIFICANT_NUM = 3
IGNORE_CASE = True


class WSD(object):
    """ Performs word sense disambiguation based on the induced word senses. """

    def __init__(self, inventory_fpath: str, language: str, verbose: bool = False,
                 skip_unknown_words: bool = True, dictionary: int = 100000):
        """ :param inventory_fpath path to a CSV file with an induced word sense inventory
            :param language code of the target language of the inventory, e.g. "en", "de" or "fr" """

        self.inventory_fpath = inventory_fpath
        self.language = language
        wv_fpath, wv_pkl_fpath = ensure_word_embeddings(self.language)
        print('Loading KeyedVectors: {}'.format(self.language))
        self._wv = KeyedVectors.load_word2vec_format(wv_fpath,
                                                     binary=False,
                                                     unicode_errors="ignore",
                                                     limit=dictionary)
        self._wv.init_sims(replace=True)  # normalize the loaded vectors to L2 norm
        print('Loading inventory: {}'.format(language))

        self._inventory = self._load_inventory()
        self._verbose = verbose
        self._unknown = Sense("UNKNOWN", "UNKNOWN", "")
        self._skip_unknown_words = skip_unknown_words

    def _load_inventory(self):
        inventory_df = read_csv(self.inventory_fpath, sep="\t", encoding="utf-8", quoting=csv.QUOTE_NONE)
        inventory_df['cluster_words'] = inventory_df.cluster.str.split(",")
        return inventory_df

    def get_senses(self, token: str, ignore_case: bool = IGNORE_CASE):
        """ Returns a list of all available senses for a given token. """
        words = {token}
        if ignore_case:
            words.add(token.title())
            words.add(token.lower())

        senses_pd = self._inventory.loc[self._inventory.word.isin(words)]
        senses_raw = list(senses_pd.itertuples(name='Row', index=False))

        senses = []
        for sense_raw in senses_raw:
            sense = Sense(sense_raw.word, sense_raw.keyword, sense_raw.cluster.split(","))
            senses.append(sense)

        return senses

    def get_best_sense_id(self, context: List[str], target_word: str, most_sign_num: int = MOST_SIGNIFICANT_NUM,
                          ignore_case: bool = IGNORE_CASE):
        """ Perform word sense disambiguation: find the correct sense of the target word inside
        the provided context.
        :param context context of the target_word that allows to disambigute its meaning, represented as a string
        :param target_word an ambigous word that need to be disambiguated
        :param most_sign_num: number of context words which are taken into account
        :param ignore_case: to look all word cases in inventory
        :return a tuple (sense_id, confidence) for the best sense """

        res = self.disambiguate_tokenized(context, target_word, most_sign_num, ignore_case)
        if len(res) > 0:
            sense, confidence = res[0]
            return sense.keyword, confidence
        else:
            return self._unknown, 1.0

    def format_result(self, token: str, sense: Tuple[Sense, float]):
        if type(sense) == str:
            sense = self._unknown

        sense_dict = {"token": token,
                      "word": sense[0].word,
                      "keyword": sense[0].keyword,
                      "cluster": sense[0].cluster,
                      "confidence": sense[1]
                      }
        return sense_dict

    def disambiguate_text(self, tokens):

        tokens_senses = []
        for token in tokens:
            token_senses = self.disambiguate_tokenized(tokens, token)
            if token_senses is None:
                token_senses = [(self._unknown, 1.0)]

            token_senses_dict = [self.format_result(token, sense) for sense in token_senses]
            tokens_senses.append(token_senses_dict)

        return tokens_senses

    def disambiguate_tokenized(self, tokens: List[str], target_word: str,
                               most_sign_num: int = MOST_SIGNIFICANT_NUM, ignore_case: bool = IGNORE_CASE):
        """ Perform word sense disambiguation: find the correct sense of the target word inside
        the provided context.
        :param tokens context of the target_word that allows to disambiguate its meaning,
        represented as a list of tokens
        :param target_word an ambigous word that need to be disambiguated
        :param most_sign_num: number of context words which are taken into account
        :param ignore_case: to look all word cases in inventory
        which are taken into account from the tokens
        :return a list of tuples (sense, confidence) """

        # get the inventory
        senses = self.get_senses(target_word, ignore_case)
        if len(senses) == 0:
            if self._verbose:
                print("Warning: word '{}' is not in the inventory. ".format(target_word))
            return [(self._unknown, 1.0)]

        # get vectors of the keywords that represent the senses
        sense_vectors = {}
        for sense in senses:
            if self._skip_unknown_words and sense.keyword not in self._wv.vocab:
                if self._verbose:
                    print("Warning: keyword '{}' is not in the word embedding model. Skipping the sense.".format(
                        sense.keyword))
            else:
                sense_vectors[sense] = self._wv[sense.keyword]

        # retrieve vectors of all context words
        context_vectors = {}
        for context_word in tokens:
            is_target = (context_word.lower().startswith(target_word.lower()) and
                         len(context_word) - len(target_word) <= 1)
            if is_target:
                continue

            if self._skip_unknown_words and context_word not in self._wv.vocab:
                if self._verbose:
                    print("Warning: context word '{}' is not in the word embedding model. Skipping the word.".format(
                        context_word))
            else:
                context_vectors[context_word] = self._wv[context_word]

        # compute distances to all prototypes for each token and pick only those which are discriminative
        context_word_scores = {}
        for context_word in context_vectors:
            scores = []
            for sense in sense_vectors:
                scores.append(context_vectors[context_word].dot(sense_vectors[sense]))

            # Could be no scores at all
            if len(scores) > 0:
                context_word_scores[context_word] = abs(max(scores) - min(scores))

        best_context_words = sorted(context_word_scores.items(), key=itemgetter(1), reverse=True)[:most_sign_num]

        # average the selected context words
        best_context_vectors = []

        for index, (context_word, _) in enumerate(best_context_words):
            best_context_vectors.append(context_vectors[context_word])

        # Could be no context vectors
        if len(best_context_vectors) == 0:
            return [(self._unknown, 1.0)]

        context_vector = mean(best_context_vectors, axis=0)

        # pick the sense which is the most similar to the context vector
        sense_scores = [(sense, float(context_vector.dot(sense_vectors[sense]))) for sense in sense_vectors]
        return sorted(sense_scores, key=itemgetter(1), reverse=True)

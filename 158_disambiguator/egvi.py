"""Dependencies required to use of this file can be installed as following:
   pip install gensim clint requests pandas nltk
   python -m nltk.downloader punkt """

import argparse
from collections import defaultdict, namedtuple
from operator import itemgetter
from os.path import exists
import os

import requests
from clint.textui import progress
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
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


def ensure_dir(f):
    """ Make the directory. """
    if not os.path.exists(f):
        os.makedirs(f)


def ensure_word_embeddings(language):
    """ Ensures that the word vectors exist or raise Exception. """

    dir_path = os.path.join("models", language)
    filename = "cc.{}.300.vec.gz".format(language)
    wv_fpath = os.path.join(dir_path, filename)
    wv_pkl_fpath = wv_fpath + ".pkl"

    if not exists(wv_fpath):
        raise Exception('No model for {} language: {}'.format(language, wv_fpath))

    return wv_fpath, wv_pkl_fpath


MOST_SIGNIFICANT_NUM = 3
IGNORE_CASE = False


class WSD(object):
    """ Performs word sense disambiguation based on the induced word senses. """

    def __init__(self, inventory_fpath, language, verbose=False, skip_unknown_words=True, dictionary=100000):
        """ :param inventory_fpath path to a CSV file with an induced word sense inventory
            :param language code of the target language of the inventory, e.g. "en", "de" or "fr" """

        wv_fpath, wv_pkl_fpath = ensure_word_embeddings(language)
        print('Loading KeyedVectors: {}'.format(language))
        self._wv = KeyedVectors.load_word2vec_format(wv_fpath,
                                                     binary=False,
                                                     unicode_errors="ignore",
                                                     limit=dictionary)
        self._wv.init_sims(replace=True)  # normalize the loaded vectors to L2 norm
        print('Loading inventory: {}'.format(language))
        self._inventory = self._load_inventory(inventory_fpath)
        self._verbose = verbose
        self._unknown = Sense("UNKNOWN", "UNKNOWN", "")
        self._skip_unknown_words = skip_unknown_words

    def _load_inventory(self, inventory_fpath):
        inventory_df = read_csv(inventory_fpath, sep="\t", encoding="utf-8")
        inventory = defaultdict(lambda: list())
        for i, row in inventory_df.iterrows():
            row_cluster = str(row.cluster)
            cluster_words = [cw.strip() for cw in row_cluster.split(",")]
            # cluster_words = [cw.strip() for cw in row.cluster.split(",")]
            inventory[row.word].append(Sense(row.word, row.keyword, cluster_words))

        return inventory

    def get_senses(self, word, ignore_case=IGNORE_CASE):
        """ Returns a list of all available senses for a given word. """

        words = set([word])
        if ignore_case:
            words.add(word.title())
            words.add(word.lower())

        senses = []
        for word in words:
            if word in self._inventory:
                senses += self._inventory[word]

        return senses

    def get_best_sense_id(self, context, target_word, most_significant_num=MOST_SIGNIFICANT_NUM,
                          ignore_case=IGNORE_CASE):
        """ Perform word sense disambiguation: find the correct sense of the target word inside
        the provided context.
        :param context context of the target_word that allows to disambigute its meaning, represented as a string
        :param target_word an ambigous word that need to be disambiguated
        :return a tuple (sense_id, confidence) for the best sense """

        res = self.disambiguate(context, target_word, most_significant_num, ignore_case)
        if len(res) > 0:
            sense, confidence = res[0]
            return sense.keyword, confidence
        else:
            return self._unknown, 1.0

    def disambiguate(self, context, target_word, most_significant_num=MOST_SIGNIFICANT_NUM, ignore_case=IGNORE_CASE):
        """ Perform word sense disambiguation: find the correct sense of the target word inside
        the provided context.
        :param context context of the target_word that allows to disambigute its meaning, represented as a string
        :param target_word an ambigous word that need to be disambiguated
        :return a list of tuples (sense, confidence) """

        if isinstance(context, str):
            try:
                # try to use nltk tokenizer
                tokens = word_tokenize(context)
            except LookupError:
                # do the simple tokenization if not installed
                tokens = context.split(" ")
        else:
            tokens = context

        return self.disambiguate_tokenized(tokens, target_word, most_significant_num, ignore_case)

    def disambiguate_tokenized(self, tokens, target_word, most_significant_num=MOST_SIGNIFICANT_NUM,
                               ignore_case=IGNORE_CASE):
        """ Perform word sense disambiguation: find the correct sense of the target word inside
        the provided context.
        :param tokens context of the target_word that allows to disambigute its meaning, represented as a list of tokens
        :param target_word an ambigous word that need to be disambiguated
        :param most_significant_num number of the most significant context words which are takein into account from the tokens
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

            # Could be no any scores
            if len(scores) > 0:
                context_word_scores[context_word] = abs(max(scores) - min(scores))

        best_context_words = sorted(context_word_scores.items(), key=itemgetter(1), reverse=True)[:most_significant_num]

        # average the selected context words
        best_context_vectors = []
        if self._verbose:
            print("Best context words for '{}' in sentence : '{}' are:".format(target_word, " ".join(tokens)))

        i = 1
        for context_word, _ in best_context_words:
            best_context_vectors.append(context_vectors[context_word])
            if self._verbose:
                print("-\t{}\t".format(i), context_word)
            i += 1

        # Could be no context vectors
        if len(best_context_vectors) == 0:
            return None

        context_vector = mean(best_context_vectors, axis=0)

        # pick the sense which is the most similar to the context vector
        sense_scores = [(sense, float(context_vector.dot(sense_vectors[sense]))) for sense in sense_vectors]
        return sorted(sense_scores, key=itemgetter(1), reverse=True)


def evaluate(wsd_model, dataset_fpath, max_context_words):
    """ Evaluates the model using the global variable wsd_model """

    output_fpath = dataset_fpath + ".filter{}.pred.csv".format(max_context_words)
    df = read_csv(dataset_fpath, sep="\t", encoding="utf-8")

    for i, row in df.iterrows():
        sense_id, _ = wsd_model.get_best_sense_id(row.context, row.word, max_context_words)
        df.loc[i, "predict_sense_id"] = sense_id

    df.to_csv(output_fpath, sep="\t", encoding="utf-8", index=False)
    print("Output:", output_fpath)

    return output_fpath


def main():
    parser = argparse.ArgumentParser(description='Sensegram egvi.')
    parser.add_argument("inventory", help="Path of the inventory file.")
    parser.add_argument("-fpath", help="Path of the file to evaluate.", required=True)
    parser.add_argument("-lang", help="Language of the stopwords.", required=True)
    parser.add_argument("-window", help="Context window size.", type=int, required=True)
    args = parser.parse_args()

    wsd_model = WSD(args.inventory, language=args.lang, verbose=True, skip_unknown_words=True)
    evaluate(wsd_model, args.fpath, args.window)


if __name__ == '__main__':
    main()

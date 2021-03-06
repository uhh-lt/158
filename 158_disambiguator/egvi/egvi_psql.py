"""Dependencies required to use of this file can be installed as following:
   pip install gensim clint requests pandas nltk
   python -m nltk.downloader punkt """

from collections import namedtuple
from operator import itemgetter

from numpy import mean
from typing import List, Tuple

from psql_server import PSQLServerModel, PSQLServerInventory

SenseBase = namedtuple('Sense', 'word keyword cluster')


class Sense(SenseBase):  # this is needed as list is an unhashable type
    def get_hash(self):
        return hash(self.word + self.keyword + "".join(self.cluster))

    def __hash__(self):
        return self.get_hash()

    def __eq__(self, other):
        return self.get_hash() == other.get_hash()


MOST_SIGNIFICANT_NUM = 5
IGNORE_CASE = True


class WSD(object):
    """ Performs word sense disambiguation based on the induced word senses. """

    def __init__(self, db_vectors: str, db_inventory: str, user: str, password: str, host: str, port: str,
                 verbose: bool = False, skip_unknown_words: bool = True):

        self.wv_vectors_db = PSQLServerModel(db=db_vectors, user=user, password=password, host=host, port=port)
        self.inventory = PSQLServerInventory(db=db_inventory, user=user, password=password, host=host, port=port)
        self._verbose = verbose
        self._unknown = (Sense("UNKNOWN", "UNKNOWN", ""), 1.0)
        self._skip_unknown_words = skip_unknown_words

    # ----------------------

    def get_context_senses(self, context_tokens: List[str], language: str, ignore_case: bool = IGNORE_CASE):
        """
        Get senses for all words in context.
        :param context_tokens: list of context tokens.
        :param language: code of the target language of the inventory, e.g. "en", "de" or "fr".
        :param ignore_case: to look all word cases in inventory
        :return: dict (word, list of senses) and list (all senses)
        """
        # Get all senses for context
        context_senses_tuples_list = self.inventory.get_tokens_senses(context_tokens, language, ignore_case)

        # Convert to list of Sense and dict of words
        context_senses_list = []
        context_senses_dict = {}
        for sense_tuple in context_senses_tuples_list:

            # list
            sense = Sense(sense_tuple[1], sense_tuple[3], sense_tuple[4].split(","))
            context_senses_list.append(sense)

            # dict
            token = sense.word.lower()
            if token not in context_senses_dict:
                context_senses_dict[token] = [sense]
            else:
                context_senses_dict[token].append(sense)

        return context_senses_dict, context_senses_list

    def get_senses_vectors(self, senses: List[Sense], language: str):
        """
        Get vectors for senses.
        :param senses: list of senses
        :param language: code of the target language of the inventory, e.g. "en", "de" or "fr".
        :return: dict (Sense, numpy vector)
        """
        # get senses of all senses' keywords
        keywords = [sense.keyword for sense in senses]
        if not keywords:
            return {}

        keywords_vectors_dict = self.wv_vectors_db.get_tokens_vectors(keywords, language)
        if not keywords_vectors_dict:
            return {}

        # match keyword vector to the sense
        senses_vectors_dict = {}
        for sense in senses:
            if sense.keyword in keywords_vectors_dict:
                senses_vectors_dict[sense] = keywords_vectors_dict[sense.keyword]

        # create dict (word - dict of senses vectors)
        token_senses_vectors_dict = {}
        for sense in senses_vectors_dict:
            token = sense.word.lower()
            if token not in token_senses_vectors_dict:
                token_senses_vectors_dict[token] = {sense: senses_vectors_dict[sense]}
            else:
                token_senses_vectors_dict[token][sense] = senses_vectors_dict[sense]

        return token_senses_vectors_dict

    @staticmethod
    def pick_best_context(context_tokens: List[str], context_vectors, target_id: int,
                          sense_vectors, most_significant_num: int):
        context_word_scores = []

        for token_index, token in enumerate(context_tokens):
            if token_index == target_id or token not in context_vectors:
                continue

            # TODO: replace by numpy matrix dot (should be faster)
            scores = []
            for sense in sense_vectors:
                scores.append(context_vectors[token].dot(sense_vectors[sense]))

            # Could be no scores at all
            if len(scores) > 0:
                context_word_scores.append((token, abs(max(scores) - min(scores))))

        context_word_scores.sort(key=itemgetter(1), reverse=True)
        best_context_words = context_word_scores[: most_significant_num]
        return best_context_words

    @staticmethod
    def average_context_vectors(best_context_words, context_vectors_dict):
        best_context_vectors = []

        for index, (context_word, _) in enumerate(best_context_words):
            best_context_vectors.append(context_vectors_dict[context_word])

        # Could be no context vectors
        if len(best_context_vectors) == 0:
            return None

        context_vector = mean(best_context_vectors, axis=0)
        return context_vector

    @staticmethod
    def format_result(token: str, sense: Tuple[Sense, float]):
        sense_dict = {"token": token,
                      "word": sense[0].word,
                      "keyword": sense[0].keyword,
                      "cluster": sense[0].cluster,
                      "confidence": sense[1]
                      }
        return sense_dict

    def disambiguate_text(self, tokens: List[str], language: str, ignore_case: bool = IGNORE_CASE,
                          most_sign_num: int = MOST_SIGNIFICANT_NUM):
        """
        Disambiguate all tokens in context.
        :param tokens: list of tokens.
        :param language: code of the target language of the inventory, e.g. "en", "de" or "fr".
        :param ignore_case: to look all word cases in inventory
        :param most_sign_num: number of context words which are taken into account
        :return: list of sorted senses with confidence for each token
        """

        # get senses inventory of all words
        token_senses_dict, context_senses_list = self.get_context_senses(tokens,
                                                                         language=language, ignore_case=ignore_case)

        # get vectors of the keywords that represent the senses (dict)
        token_senses_vectors_dict = self.get_senses_vectors(context_senses_list, language=language)

        # retrieve vectors of all context words
        if token_senses_vectors_dict:
            context_vectors_dict = self.wv_vectors_db.get_tokens_vectors(tokens, lang=language)
        else:
            context_vectors_dict = {}

        result = []

        for token_index, token in enumerate(tokens):
            if token.lower() not in token_senses_vectors_dict:
                sense = self._unknown
                sense_dict = self.format_result(token, sense)
                token_senses = [sense_dict]
                result.append(token_senses)
                continue

            token_senses_vectors_list = token_senses_vectors_dict[token.lower()]

            # compute distances to all prototypes for each token and pick only those which are discriminative
            best_context_words = self.pick_best_context(tokens, context_vectors_dict, token_index,
                                                        token_senses_vectors_list, most_sign_num)

            if best_context_words:
                # average the selected context words
                context_vector = self.average_context_vectors(best_context_words, context_vectors_dict)

                # pick the sense which is the most similar to the context vector
                sense_scores = [(sense, float(context_vector.dot(token_senses_vectors_list[sense])))
                                for sense in token_senses_vectors_list]
            # if there is no context for word - any sense is possible
            else:
                sense_scores = [(sense, 1.0 / len(token_senses_vectors_list))
                                for sense in token_senses_vectors_list]

            senses_sorted = sorted(sense_scores, key=itemgetter(1), reverse=True)

            if senses_sorted is None:
                senses_sorted = self._unknown

            # Format results
            token_senses = []
            for sense in senses_sorted:
                sense_dict = self.format_result(token, sense)
                token_senses.append(sense_dict)

            result.append(token_senses)

        return result

    def disambiguate_word(self, tokens: List[str], target_word: str, language: str,
                          ignore_case=IGNORE_CASE, most_sign_num=MOST_SIGNIFICANT_NUM):
        """
        Disambiguate single token in context.
        :param tokens: list of tokens.
        :param target_word: word to disambiguate
        :param language: code of the target language of the inventory, e.g. "en", "de" or "fr".
        :param ignore_case: to look all word cases in inventory
        :param most_sign_num: number of context words which are taken into account
        :return: list of sorted senses with confidence for target token
        """
        # get senses inventory of all words
        token_senses_dict, context_senses_list = self.get_context_senses(tokens,
                                                                         language=language, ignore_case=ignore_case)

        # get vectors of the keywords that represent the senses (dict)
        token_senses_vectors_dict = self.get_senses_vectors(context_senses_list, language=language)

        # retrieve vectors of all context words
        context_vectors_dict = self.wv_vectors_db.get_tokens_vectors(tokens, lang=language)

        result = []

        if target_word.lower() not in token_senses_vectors_dict:
            sense = self._unknown
            sense_dict = self.format_result(target_word, sense)
            token_senses = [sense_dict]
            result.append(token_senses)
            return result

        token_senses_vectors_dict = token_senses_vectors_dict[target_word.lower()]

        # Find index of target
        target_id = tokens.index(target_word)

        # compute distances to all prototypes for each token and pick only those which are discriminative
        best_context_words = self.pick_best_context(tokens, context_vectors_dict, target_id,
                                                    token_senses_vectors_dict, most_sign_num)

        # average the selected context words
        context_vector = self.average_context_vectors(best_context_words, context_vectors_dict)

        # pick the sense which is the most similar to the context vector
        sense_scores = [(sense, float(context_vector.dot(token_senses_vectors_dict[sense])))
                        for sense in token_senses_vectors_dict]
        senses_sorted = sorted(sense_scores, key=itemgetter(1), reverse=True)

        if senses_sorted is None:
            senses_sorted = [self._unknown]

        # Format results
        token_senses = []
        for sense in senses_sorted:
            sense_dict = self.format_result(target_word, sense)
            token_senses.append(sense_dict)

        result.append(token_senses)

        return result

    def disambiguate_word_by_id(self, tokens: List[str], target_id: int, language: str,
                                ignore_case=IGNORE_CASE, most_sign_num=MOST_SIGNIFICANT_NUM):
        """
        Disambiguate single token in context.
        :param tokens: list of tokens.
        :param target_id: id of target word in context
        :param language: code of the target language of the inventory, e.g. "en", "de" or "fr".
        :param ignore_case: to look all word cases in inventory
        :param most_sign_num: number of context words which are taken into account
        :return: list of sorted senses with confidence for target token
        """
        # get senses inventory of all words
        token_senses_dict, context_senses_list = self.get_context_senses(tokens,
                                                                         language=language, ignore_case=ignore_case)

        # get vectors of the keywords that represent the senses (dict)
        token_senses_vectors_dict = self.get_senses_vectors(context_senses_list, language=language)

        # retrieve vectors of all context words
        context_vectors_dict = self.wv_vectors_db.get_tokens_vectors(tokens, lang=language)

        result = []

        token = tokens[target_id]

        if token.lower() not in token_senses_vectors_dict:
            sense = self._unknown
            sense_dict = self.format_result(token, sense)
            token_senses = [sense_dict]
            result.append(token_senses)
            return result

        token_senses_vectors_dict = token_senses_vectors_dict[token.lower()]

        # compute distances to all prototypes for each token and pick only those which are discriminative
        best_context_words = self.pick_best_context(tokens, context_vectors_dict, target_id,
                                                    token_senses_vectors_dict, most_sign_num)

        # average the selected context words
        context_vector = self.average_context_vectors(best_context_words, context_vectors_dict)

        # pick the sense which is the most similar to the context vector
        sense_scores = [(sense, float(context_vector.dot(token_senses_vectors_dict[sense])))
                        for sense in token_senses_vectors_dict]
        senses_sorted = sorted(sense_scores, key=itemgetter(1), reverse=True)

        if senses_sorted is None:
            senses_sorted = [self._unknown]

        # Format results
        token_senses = []
        for sense in senses_sorted:
            sense_dict = self.format_result(token, sense)
            token_senses.append(sense_dict)

        result.append(token_senses)

        return result

    def get_senses(self, word, language: str, ignore_case=IGNORE_CASE):
        """ Returns a list of all available senses for a given word. """
        rows = self.inventory.get_word_senses(word, language, ignore_case)

        senses = []
        for row in rows:
            senses.append(Sense(row[1], row[3], row[4].split(",")))

        return senses

    def get_best_sense_id(self, context, target_word: str, language: str,
                          most_sign_num=MOST_SIGNIFICANT_NUM, ignore_case=IGNORE_CASE):
        """ Perform word sense disambiguation: find the correct sense of the target word inside
        the provided context.
        :param context context of the target_word that allows to disambiguate its meaning, represented as a string
        :param language: code of the target language of the inventory, e.g. "en", "de" or "fr".
        :param target_word an ambigous word that need to be disambiguated
        :param most_sign_num: number of context words which are taken into account
        :param ignore_case: to look all word cases in inventory
        :return a tuple (sense_id, confidence) for the best sense """

        res = self.disambiguate_word(context, target_word, language, ignore_case, most_sign_num)
        if len(res) > 0:
            sense, confidence = res[0]
            return sense.keyword, confidence
        else:
            return self._unknown

import numpy as np
from typing import List
import psycopg2


class PSQLServer(object):
    def __init__(self, db: str, user: str, password: str, host: str, port: str):
        """:param db: database name
           :return: Connection object or None"""

        self.conn = psycopg2.connect(database=db,
                                     user=user,
                                     password=password,
                                     host=host,
                                     port=port)

    def sql_select(self, query: str, input_tuple=None):
        cur = self.conn.cursor()
        if input_tuple:
            cur.execute(query, (input_tuple,))
        else:
            cur.execute(query)
        rows = cur.fetchall()
        return rows

    def get_vocab(self, lang):
        table_name = lang + "_"
        query = "SELECT word FROM {}_".format(table_name)
        rows = self.sql_select(query)
        vocab = set([item for sublist in rows for item in sublist])
        return vocab


class PSQLServerModel(PSQLServer):
    """Create a connection to the SQLite database with word vectors."""

    def get_word_vector(self, lang: str, word: str):
        word = word.replace('"', '')
        table_name = lang + "_"

        # Make tuple from a single variable to put into psql query
        word_tuple = (word,)

        query = """SELECT * FROM {table} WHERE word = %s""".format(table=table_name)
        rows = self.sql_select(query, word_tuple)
        if len(rows) == 0:
            word_vector = None
        else:
            word_vector = np.array(rows[0][1:])
        return word_vector

    def get_tokens_vectors(self, words: List[str], lang: str):
        words = tuple([word.replace('"', '') for word in words])
        table_name = lang + "_"
        query = """SELECT * FROM {table} WHERE word in %s""".format(table=table_name)
        rows = self.sql_select(query, words)
        if len(rows) == 0:
            vectors_dict = None
        else:
            vectors_dict = dict()
            for row in rows:
                vectors_dict[row[0]] = np.array(row[1:])
        return vectors_dict


class PSQLServerInventory(PSQLServer):
    """Create a connection to the SQLite database with language inventories."""

    def get_tokens_senses(self, tokens: List[str], lang: str, ignore_case: bool):
        table_name = lang + "_"

        if ignore_case:
            tokens_new = []
            for token in tokens:
                tokens_new.append(token)
                tokens_new.append(token.title())
                tokens_new.append(token.lower())
            tokens_tuple = tuple(tokens_new)
        else:
            tokens_tuple = tuple(tokens)

        query = """SELECT * FROM {table} WHERE word in %s""".format(table=table_name)
        rows = self.sql_select(query, tokens_tuple)
        return rows

    def get_word_senses(self, word: str, lang: str, ignore_case: bool):
        table_name = lang + "_"
        word = word.replace('"', '')

        if ignore_case:
            words = tuple({word, word.title(), word.lower()})
            query = """SELECT * FROM {table} WHERE word in %s""".format(table=table_name)
            rows = self.sql_select(query, words)
        else:
            query = """SELECT * FROM {table} WHERE word = %s""".format(table=table_name)
            word_tuple = (word,)
            rows = self.sql_select(query, word_tuple)

        return rows

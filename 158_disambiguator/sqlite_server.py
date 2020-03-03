import sqlite3
from sqlite3 import Error
import numpy as np
from typing import List


class SqliteServer(object):
    def __init__(self, db, table_name: str):
        """:param db: database name
           :return: Connection object or None"""
        self.table_name = table_name
        self.db = db
        conn = self.create_connection()
        conn.close()
        self.vocab = self.__get_vocab__()

    def create_connection(self):
        """ create a database connection to the SQLite database
        :return: Connection object or None
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db)
        except Error as e:
            print(e)

        return conn

    def sql_query(self, query: str):
        conn = self.create_connection()
        c = conn.cursor()
        c.execute(query)
        rows = c.fetchall()
        conn.close()
        return rows

    def __get_vocab__(self):
        query = "SELECT word FROM {}".format(self.table_name)
        rows = self.sql_query(query)
        vocab = set([item for sublist in rows for item in sublist])
        return vocab


class SqliteServerModel(SqliteServer):
    """Create a connection to the SQLite database with word vectors."""

    def __init__(self, db, lang: str):
        table_name = lang + "_"
        super().__init__(db, table_name=table_name)

    @staticmethod
    def _chunks_(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def get_tokens_vectors(self, tokens: List[str]):
        # Sqlite won't accept such long query
        # Split tokens to batches
        if len(tokens) > 20:
            tokens_batches = list(SqliteServerModel._chunks_(tokens, 20))
            result_dict = {}
            for tokens_batch in tokens_batches:
                vectors_batch = self._get_tokens_vectors_batch_(tokens_batch)
                result_dict.update(vectors_batch)
        else:
            result_dict = self._get_tokens_vectors_batch_(tokens)

        return result_dict

    def _get_tokens_vectors_batch_(self, tokens: List[str]):
        query = 'SELECT * FROM {table} WHERE '.format(table=self.table_name)
        for index, token in enumerate(tokens):
            if token is not None:
                token_clear = token.replace('"', '')
                query += 'word = "{word}"'.format(word=token_clear)
                if index + 1 != len(tokens):
                    query += ' OR '

        rows = self.sql_query(query)
        vectors_dict = {}
        for row in rows:
            vectors_dict[row[0]] = np.array(row[1:])
        return vectors_dict

    def get_word_vector(self, word: str):
        word = word.replace('"', '')
        query = 'SELECT * FROM {table} WHERE word = "{word}"'.format(table=self.table_name, word=word)
        rows = self.sql_query(query)
        word_vector = np.array(rows[0][1:])
        return word_vector


class SqliteServerInventory(SqliteServer):
    """Create a connection to the SQLite database with language inventories."""

    def __init__(self, db, lang: str):
        table_name = lang + "_"
        super().__init__(db, table_name=table_name)

    def get_tokens_senses(self, tokens: List[str], ignore_case: bool):
        query = 'SELECT * FROM {table} WHERE '.format(table=self.table_name)
        for index, token in enumerate(tokens):
            token_clear = token.replace('"', '')
            query += 'word = "{word}"'.format(word=token_clear)
            if ignore_case:
                query += ' OR word = "{word}"'.format(word=token_clear.title())
                query += ' OR word = "{word}"'.format(word=token_clear.lower())
                query += ' OR word = "{word}"'.format(word=token_clear.upper())

            if index + 1 != len(tokens):
                query += ' OR '

        rows = self.sql_query(query)
        return rows

    def get_word_senses(self, word: str):
        word = word.replace('"', '')
        query = 'SELECT * FROM {table} WHERE word = "{word}"'.format(table=self.table_name, word=word)
        rows = self.sql_query(query)
        return rows

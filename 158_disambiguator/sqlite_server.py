import sqlite3
from sqlite3 import Error
import numpy as np


class SqliteServer(object):
    def __init__(self, db, lang):
        """:param db: database name
           :return: Connection object or None"""
        self.lang = lang
        self.db = db
        conn = self.create_connection()
        print('SQLite init succeed')
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
        query = "SELECT word FROM {}".format(self.lang)
        rows = self.sql_query(query)
        vocab = set([item for sublist in rows for item in sublist])
        return vocab


class SqliteServerModel(SqliteServer):
    """Create a connection to the SQLite database with word vectors."""

    def __init__(self, db, lang):
        super().__init__(db, lang)

    def get_word_vector(self, word):
        query = "SELECT * FROM {table} WHERE word = '{word}'".format(table=self.lang, word=word)
        rows = self.sql_query(query)
        word_vector = np.array(rows[0][1:])
        return word_vector


class SqliteServerInventory(SqliteServer):
    """Create a connection to the SQLite database with language inventories."""

    def __init__(self, db, lang):
        super().__init__(db, lang)

    def get_word_senses(self, word):
        query = "SELECT * FROM {table} WHERE word = '{word}'".format(table=self.lang, word=word)
        rows = self.sql_query(query)
        return rows

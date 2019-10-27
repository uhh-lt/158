import sqlite3
from sqlite3 import Error
import numpy as np


class SqliteServer(object):
    """Create a connection to the SQLite database with word vectors."""

    def __init__(self, db, lang):
        """:param db: database name
           :return: Connection object or None"""
        self.conn = None
        self.lang = lang
        try:
            self.conn = sqlite3.connect(db)
        except Error as e:
            print(e)
        else:
            print('SQLite connection succeed')
            self.vocab = self.__get_vocab__()

    def __get_vocab__(self):
        c = self.conn.cursor()
        c.execute("SELECT WORD FROM {}".format(self.lang))
        rows = c.fetchall()
        c.close()
        vocab = [item for sublist in rows for item in sublist]
        return vocab

    def get_word_vector(self, word):
        c = self.conn.cursor()
        sql_query = "SELECT * FROM {table} WHERE word = '{word}'".format(table=self.lang, word=word)
        c.execute(sql_query)
        rows = c.fetchall()
        c.close()
        word_vector = np.array(rows[0][1:])
        return word_vector

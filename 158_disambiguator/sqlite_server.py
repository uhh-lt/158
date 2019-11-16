import sqlite3
from sqlite3 import Error
import numpy as np


class SqliteServer(object):
    def __init__(self, db, table_name):
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

    def __init__(self, db, lang):
        table_name = lang + "_"
        super().__init__(db, table_name=table_name)

    def get_word_vector(self, word):
        query = "SELECT * FROM {table} WHERE word = '{word}'".format(table=self.table_name, word=word)
        rows = self.sql_query(query)
        word_vector = np.array(rows[0][1:])
        return word_vector


class SqliteServerInventory(SqliteServer):
    """Create a connection to the SQLite database with language inventories."""

    def __init__(self, db, lang):
        table_name = lang + "_"
        super().__init__(db, table_name=table_name)

    def get_word_senses(self, word):
        query = "SELECT * FROM {table} WHERE word = '{word}'".format(table=self.table_name, word=word)
        rows = self.sql_query(query)
        return rows

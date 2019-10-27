import sqlite3
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sqlite3 import Error


def load_keyed_vectors(wv_fpath, limit=100000):
    wv = KeyedVectors.load_word2vec_format(wv_fpath, binary=False, unicode_errors="ignore", limit=limit)
    wv.init_sims(replace=True)  # normalize the loaded vectors to L2 norm
    return wv


def create_vectors_df(wv):
    vectors_df = pd.DataFrame(wv.vectors)
    vectors_df["word"] = wv.index2word
    vectors_df.set_index('word', inplace=True)
    return vectors_df


# def dataframe_test(vectors_df, model):
#     vector_from_df = np.array(vectors_df.loc[','].tolist())
#     vector_from_model = wv[',']
#     assert np.array_equal(vector_from_df, vector_from_model) is True


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print('Connection succeed')
    except Error as e:
        print(e)

    return conn


def upload_vectors_sqlite(vectors: pd.DataFrame, database: str, table_name: str):
    conn = create_connection(database)
    vectors.to_sql(table_name, conn, if_exists='replace', index=True)
    conn.close()
    print('Upload succeed: {}'.format(table_name))


def main():

    sqlite_db = "./models/Vectors.db"
    langs = ['ru', 'en']

    for lang in langs:
        print('Start: {}'.format(lang))
        wv_fpath = "./models/{lang}/cc.{lang}.300.vec.gz".format(lang=lang)
        print('Loading model: {}'.format(lang))
        wv = load_keyed_vectors(wv_fpath, limit=100000)
        print('Creating dataframe: {}'.format(lang))
        vectors_df = create_vectors_df(wv)  # Create df from vectors
        del wv
        print('Uploading to sqlite: {}'.format(lang))
        upload_vectors_sqlite(vectors_df, database=sqlite_db, table_name=lang)  # Create sqlite database
    print('Finish')


if __name__ == '__main__':
    main()

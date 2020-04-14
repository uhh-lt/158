import os
import pandas as pd
import logging
from gensim.models import KeyedVectors
from sqlalchemy import create_engine

PSQL_USER = "158_user"
PSQL_PASSWORD = "158"
PSQL_DB = "fasttext_vectors"
PSQL_IP = "localhost"
PSQL_PORT = "10153"

LIMIT = 100
FASTTEXT_PATH = "./fasttext_models/{lang}/cc.{lang}.300.vec.gz"

logging.basicConfig(filename="vectors_psql.log", level=logging.INFO, filemode='w')


def load_keyed_vectors(wv_fpath, limit):
    wv = KeyedVectors.load_word2vec_format(wv_fpath, binary=False, unicode_errors="ignore", limit=limit)
    wv.init_sims(replace=True)  # normalize the loaded vectors to L2 norm
    return wv


def create_vectors_df(wv):
    vectors_df = pd.DataFrame(wv.vectors)
    vectors_df["word"] = wv.index2word
    vectors_df.set_index('word', inplace=True)
    return vectors_df


def upload_vectors_sqlite(vectors: pd.DataFrame, database: str, table_name: str):
    engine = create_engine('postgresql://{user}:{pswd}@{ip}:{port}/{db}'.format(user=PSQL_USER,
                                                                                pswd=PSQL_PASSWORD,
                                                                                ip=PSQL_IP,
                                                                                port=PSQL_PORT,
                                                                                db=database))
    vectors.to_sql(table_name, engine, if_exists='replace')
    print('Upload succeed: {}'.format(table_name))
    logging.info('Upload succeed: {}'.format(table_name))


def main():
    lang_list = ['af', 'als', 'am', 'an', 'ar', 'arz', 'as',
                 'ast', 'az', 'azb', 'ba', 'bar', 'bcl', 'be',
                 'bg', 'bh', 'bn', 'bo', 'bpy', 'br', 'bs',
                 'ca', 'ce', 'ceb', 'ckb', 'co', 'cs', 'cv',
                 'cy', 'da', 'de', 'diq', 'dv', 'el', 'eml',
                 'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi',
                 'fr', 'frr', 'fy', 'ga', 'gd', 'gl', 'gom',
                 'gu', 'gv', 'he', 'hi', 'hif', 'hr', 'hsb',
                 'ht', 'hu', 'hy', 'ia', 'id', 'ilo', 'io',
                 'is', 'it', 'ja', 'jv', 'ka', 'kk', 'km',
                 'kn', 'ku', 'ky', 'la', 'lb', 'li', 'lmo',
                 'lt', 'lv', 'mai', 'mg', 'mhr', 'min', 'mk',
                 'ml', 'mn', 'mr', 'mrj', 'ms', 'mt', 'mwl',
                 'my', 'myv', 'mzn', 'nah', 'nap', 'nds', 'ne',
                 'new', 'nl', 'nn', 'no', 'nso', 'oc', 'or',
                 'os', 'pa', 'pam', 'pfl', 'pl', 'pms', 'pnb',
                 'ps', 'pt', 'qu', 'rm', 'ro', 'ru', 'sa',
                 'sah', 'sc', 'scn', 'sco', 'sd', 'sh', 'si',
                 'sk', 'sl', 'so', 'sq', 'sr', 'su', 'sv',
                 'sw', 'ta', 'te', 'tg', 'th', 'tk', 'tl',
                 'tr', 'tt', 'ug', 'uk', 'ur', 'uz', 'vec',
                 'vi', 'vls', 'vo', 'wa', 'war', 'xmf', 'yi',
                 'yo', 'zea', 'zh', 'ko']

    for lang in lang_list:
        print('Start: {}'.format(lang))
        logging.info('Start: {}'.format(lang))
        wv_fpath = FASTTEXT_PATH.format(lang=lang)
        if not os.path.exists(wv_fpath):
            logging.error('No model for {lang}'.format(lang=lang))
            print('No model for {lang}'.format(lang=lang))
            continue

        print('Loading model: {}'.format(lang))
        logging.info('Loading model: {}'.format(lang))
        wv = load_keyed_vectors(wv_fpath, limit=LIMIT)
        print('Creating dataframe: {}'.format(lang))
        logging.info('Creating dataframe: {}'.format(lang))
        vectors_df = create_vectors_df(wv)  # Create df from vectors
        del wv
        print('Uploading to psql: {}'.format(lang))
        logging.info('Uploading to psql: {}'.format(lang))
        table_name = lang + "_"
        upload_vectors_sqlite(vectors_df, database=PSQL_DB, table_name=table_name)  # Create sqlite database
    print('Finish')


if __name__ == '__main__':
    main()

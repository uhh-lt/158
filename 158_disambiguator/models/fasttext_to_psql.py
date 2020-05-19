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

LIMIT = 100000
FASTTEXT_PATH = "./fasttext_models/{lang}/cc.{lang}.300.vec.gz"

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/fasttext_psql.log"),
        logging.StreamHandler()
    ]
)


def load_keyed_vectors(wv_fpath, limit):
    wv = KeyedVectors.load_word2vec_format(wv_fpath, binary=False, unicode_errors="ignore", limit=limit)
    wv.init_sims(replace=True)  # Normalize the loaded vectors to L2 norm
    return wv


def create_vectors_df(wv):
    vectors_df = pd.DataFrame(wv.vectors)
    vectors_df["word"] = wv.index2word
    vectors_df.set_index('word', inplace=True)
    return vectors_df


def upload_vectors_sqlite(vectors: pd.DataFrame, database: str, table_name: str):
    url = 'postgresql://{user}:{pswd}@{ip}:{port}/{db}'.format(user=PSQL_USER,
                                                               pswd=PSQL_PASSWORD,
                                                               ip=PSQL_IP,
                                                               port=PSQL_PORT,
                                                               db=database)
    engine = create_engine(url)
    vectors.to_sql(table_name, engine, if_exists='replace')
    logging.info('Upload succeed: {}'.format(table_name))
    return None


def load_lang(lang: str, fasttext_path):
    logging.info('Start: {}'.format(lang))

    if not os.path.exists(fasttext_path):
        logging.error('No model for {lang}'.format(lang=lang))
        return None

    logging.info('Loading model: {}'.format(lang))
    wv = load_keyed_vectors(fasttext_path, limit=LIMIT)

    logging.info('Creating dataframe: {}'.format(lang))
    vectors_df = create_vectors_df(wv)

    logging.info('Uploading to psql: {}')
    table_name = lang + "_"
    upload_vectors_sqlite(vectors_df, database=PSQL_DB, table_name=table_name)
    return None


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
        fasttext_fpath = FASTTEXT_PATH.format(lang=lang)
        load_lang(lang, fasttext_fpath)

    logging.info('Finish')


if __name__ == '__main__':
    main()

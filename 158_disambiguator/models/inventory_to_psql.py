import pandas as pd
import os
import logging
from sqlalchemy import create_engine

PSQL_USER = "158_user"
PSQL_PASSWORD = "158"
PSQL_DB = "inventory"
PSQL_IP = "localhost"
PSQL_PORT = "10153"

INVENTORY_PATH = "./inventories/{lang}/cc.{lang}.300.vec.gz.top{knn}.inventory.tsv.gz"
NEIGHBORS = 200

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/inventory_psql.log"),
        logging.StreamHandler()
    ]
)


def load_inventory(inventory_fpath):
    inventory_df = pd.read_csv(inventory_fpath,
                               compression='gzip',
                               sep="\t",
                               encoding="utf-8",
                               quoting=3,
                               error_bad_lines=False)
    return inventory_df


def create_vectors_df(wv):
    vectors_df = pd.DataFrame(wv.vectors)
    vectors_df["word"] = wv.index2word
    vectors_df.set_index('word', inplace=True)
    return vectors_df


def upload_vectors_sqlite(inventory_df: pd.DataFrame, database: str, table_name: str):
    engine = create_engine('postgresql://{user}:{pswd}@{ip}:{port}/{db}'.format(user=PSQL_USER,
                                                                                pswd=PSQL_PASSWORD,
                                                                                ip=PSQL_IP,
                                                                                port=PSQL_PORT,
                                                                                db=database))
    inventory_df.to_sql(table_name, engine, if_exists='replace')
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
        logging.info('Start: {}'.format(lang))
        inventory_lang_path = INVENTORY_PATH.format(lang=lang, knn=NEIGHBORS)
        if not os.path.exists(inventory_lang_path):
            log_error_message = 'No inventory for {lang} with {knn} neighbors'.format(lang=lang, knn=NEIGHBORS)
            logging.error(log_error_message)
            continue

        logging.info('Loading inventory: {}'.format(lang))
        inventory_df = load_inventory(inventory_lang_path)

        logging.info('Uploading to psql: {}'.format(lang))
        table_name = lang + "_"
        upload_vectors_sqlite(inventory_df, database=PSQL_DB, table_name=table_name)

    logging.info('Finish')


if __name__ == '__main__':
    main()

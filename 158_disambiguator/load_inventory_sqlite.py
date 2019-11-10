import sqlite3
import pandas as pd
import os
from sqlite3 import Error


def load_inventory(inventory_fpath):
    inventory_df = pd.read_csv(inventory_fpath, sep="\t", encoding="utf-8")
    return inventory_df


def create_vectors_df(wv):
    vectors_df = pd.DataFrame(wv.vectors)
    vectors_df["word"] = wv.index2word
    vectors_df.set_index('word', inplace=True)
    return vectors_df


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


def upload_inventory_sqlite(inventory: pd.DataFrame, database: str, table_name: str):
    conn = create_connection(database)
    inventory.to_sql(table_name, conn, if_exists='replace', index=True)
    conn.close()
    print('Upload succeed: {}'.format(table_name))


def main():
    sqlite_db = "./models/Inventory.db"
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
    neighbors = 200

    inventory_path = "./models/inventories/{lang}/cc.{lang}.300.vec.gz.top{knn}.inventory.tsv"
    for lang in lang_list:
        print('Start: {}'.format(lang))
        inventory_lang_path = inventory_path.format(lang=lang, knn=neighbors)
        if not os.path.exists(inventory_lang_path):
            print('No inventory for {lang} with {knn} neighbors'.format(lang=lang, knn=neighbors))
            continue

        print('Loading inventory: {}'.format(lang))
        inventory_df = load_inventory(inventory_lang_path)
        print('Uploading to sqlite: {}'.format(lang))
        upload_inventory_sqlite(inventory_df, database=sqlite_db, table_name=lang)  # Create sqlite database
    print('Finish')


if __name__ == '__main__':
    main()

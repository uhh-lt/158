import os
import gzip
import codecs
import string
import logging
import argparse
from time import time
from glob import glob
from collections import Counter
from traceback import format_exc
from os.path import join, exists
from typing import List, Dict, Set

import faiss
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx import Graph
from pandas import read_csv
from gensim.models import KeyedVectors
from chinese_whispers import chinese_whispers, aggregate_clusters

from load_fasttext import download_word_embeddings

wsi_data_dir = "/home/panchenko/russe-wsi-full/data/"

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Max number of neighbors
verbose = True
LIMIT = 100000
BATCH_SIZE = 2000
GPU_DEVICE = 0

try:
    wv
except NameError:
    wv = None


def get_embedding_path(language):
    """ Ensures that the word vectors exist by downloading them if needed. """

    dir_path = os.path.join("fasttext_models", language)
    wv_fpath = os.path.join(dir_path, "cc.{}.300.vec.gz".format(language))

    if os.path.exists(wv_fpath):
        print('Embedding for {} exists'.format(language))
    else:
        print('Embedding for {} does not exist, loading'.format(language))
        download_word_embeddings(language)

    wv_pkl_fpath = wv_fpath + ".pkl"

    return wv_fpath, wv_pkl_fpath


def get_ru_wsi_vocabulary() -> Set:
    dataset_names = ["active-dict", "wiki-wiki", "bts-rnc"]

    voc = set(["ключ", "замок", "коса"])

    for dataset_name in dataset_names:
        train_fpath = join(join(wsi_data_dir, dataset_name), "train.csv")
        test_fpath = join(join(wsi_data_dir, dataset_name), "test.csv")

        if exists(train_fpath):
            train = read_csv(train_fpath, sep="\t", encoding="utf-8")

        if exists(test_fpath):
            test = read_csv(test_fpath, sep="\t", encoding="utf-8")

        for i, row in test.iterrows():
            voc.add(row.word)
        for i, row in train.iterrows():
            voc.add(row.word)

    return voc


def save_to_gensim_format(wv, output_fpath: str):
    tic = time()
    wv.save(output_fpath)
    print("Saved in {} sec.".format(time() - tic))


def load_globally(word_vectors_fpath: str, faiss_gpu: bool):
    global wv
    global index_faiss

    if not wv:
        print("Loading word vectors from:", word_vectors_fpath)
        tic = time()
        if word_vectors_fpath.endswith(".vec.gz"):
            wv = KeyedVectors.load_word2vec_format(word_vectors_fpath, binary=False, unicode_errors="ignore")
        else:
            wv = KeyedVectors.load(word_vectors_fpath)
        print("Loaded in {} sec.".format(time() - tic))
    else:
        print("Using loaded word vectors.")

    wv.init_sims(replace=True)

    if faiss_gpu:
        res = faiss.StandardGpuResources()  # use a single GPU
        index_flat = faiss.IndexFlatIP(wv.vector_size)  # build a flat (CPU) index
        index_faiss = faiss.index_cpu_to_gpu(res, GPU_DEVICE, index_flat)  # make it into a gpu index
        index_faiss.add(wv.syn0norm)  # add vectors to the index
    else:
        index_faiss = faiss.IndexFlatIP(wv.vector_size)
        index_faiss.add(wv.syn0norm)
    return wv


def get_nns(target: str, neighbors_number: int):
    """
    Get neighbors for target word
    :param target: word to find neighbors
    :param neighbors_number: number of neighbors
    :return: list of target neighbors
    """
    target_neighbors = voc_neighbors[target]
    if len(target_neighbors) >= neighbors_number:
        return target_neighbors[:neighbors_number]
    else:
        print("neighbors_number {} is more than precomputed {}".format(neighbors_number, len(target_neighbors)))
        exit(1)


def get_nns_faiss_batch(targets: List, batch_size: int, neighbors_number: int = 200) -> Dict:
    """
    Get neighbors for targets by Faiss with a batch-split.
    :param targets: list of target words
    :param batch_size: how many words to push into Faiss
    :param neighbors_number: number of neighbors
    :return: dict of word -> list of neighbors
    """

    word_neighbors_dict = dict()
    if verbose:
        print("Start Faiss with batches")

    logger_info.info("Start Faiss with batches")

    for start in range(0, len(targets), batch_size):
        end = start + batch_size

        if verbose:
            print("batch {} to {} of {}".format(start, end, len(targets)))

        logger_info.info("batch {} to {} of {}".format(start, end, len(targets)))

        batch_dict = get_nns_faiss(targets[start:end], neighbors_number=neighbors_number)
        word_neighbors_dict = {**word_neighbors_dict, **batch_dict}

    return word_neighbors_dict


def get_nns_faiss(targets: List, neighbors_number: int = 200) -> Dict:
    """
    Get nearest neighbors for list of targets without batches.
    :param targets: list of target words
    :param neighbors_number: number of neighbors
    :return: dict of word -> list of neighbors
    """

    numpy_vec = np.array([wv[target] for target in targets])  # Create array of batch vectors
    D, I = index_faiss.search(numpy_vec, neighbors_number + 1)  # Find neighbors

    # Write neighbors into dict
    word_neighbors_dict = dict()
    for word_index, (_D, _I) in enumerate(zip(D, I)):

        # Check if word is punct
        word = targets[word_index]
        nns_list = []
        for n, (d, i) in enumerate(zip(_D.ravel(), _I.ravel())):
            if n > 0:
                nns_list.append((wv.index2word[i], d))

        word_neighbors_dict[word] = nns_list

    return word_neighbors_dict


def in_nns(nns, word: str) -> bool:
    """Check if word is in list of tuples nns."""
    for w, s in nns:
        if word.strip().lower() == w.strip().lower():
            return True

    return False


def get_pair(first, second) -> tuple:
    pair_lst = sorted([first, second])
    sorted_pair = (pair_lst[0], pair_lst[1])
    return sorted_pair


def get_disc_pairs(ego, neighbors_number: int) -> Set:
    pairs = set()

    nns = get_nns(ego, neighbors_number)
    nns_words = [row[0] for row in nns]  # list of neighbors (only words)
    wv_neighbors = np.array([wv[nns_word] for nns_word in nns_words])
    wv_ego = np.array(wv[ego])
    wv_negative_neighbors = (wv_neighbors - wv_ego) * (-1)  # untop vectors

    D, I = index_faiss.search(wv_negative_neighbors, 1 + 1)  # find top neighbor for each difference

    # Write down top-untop pairs
    pairs_list_2 = list()
    for word_index, (_D, _I) in enumerate(zip(D, I)):
        for n, (d, i) in enumerate(zip(_D.ravel(), _I.ravel())):
            if wv.index2word[i] != ego:  # faiss find either ego-word or untop we need
                pairs_list_2.append((nns_words[word_index], wv.index2word[i]))
                break

    # Filter pairs
    for pair in pairs_list_2:
        if in_nns(nns, pair[1]):
            pairs.add(get_pair(pair[0], pair[1]))

    return pairs


def get_nodes(pairs: Set) -> Counter:
    nodes = Counter()
    for src, dst in pairs:
        nodes.update([src])
        nodes.update([dst])

    return nodes


def list2dict(lst: list) -> Dict:
    return {p[0]: p[1] for p in lst}


def wsi(ego, neighbors_number: int) -> Dict:
    """
    Gets graph of neighbors for word (ego)
    :param ego: word
    :param neighbors_number: number of neighbors
    :return: dict of network and nodes
    """
    tic = time()
    ego_network = Graph(name=ego)

    pairs = get_disc_pairs(ego, neighbors_number)
    nodes = get_nodes(pairs)

    ego_network.add_nodes_from([(node, {'size': size}) for node, size in nodes.items()])

    for r_node in ego_network:
        related_related_nodes = list2dict(get_nns(r_node, neighbors_number))
        related_related_nodes_ego = sorted(
            [(related_related_nodes[rr_node], rr_node) for rr_node in related_related_nodes if rr_node in ego_network],
            reverse=True)[:neighbors_number]

        related_edges = []
        for w, rr_node in related_related_nodes_ego:
            if get_pair(r_node, rr_node) not in pairs:
                related_edges.append((r_node, rr_node, {"weight": w}))
            else:
                # print("Skipping:", r_node, rr_node)
                pass
        ego_network.add_edges_from(related_edges)

    chinese_whispers(ego_network, weighting="top", iterations=20)
    if verbose:
        print("{}\t{:f} sec.".format(ego, time() - tic))

    return {"network": ego_network, "nodes": nodes}


def draw_ego(G, show=False, save_fpath=""):
    label2id = {}
    colors = []
    sizes = []
    for node in G.nodes():
        label = G.node[node]['label']
        if label not in label2id:
            label2id[label] = len(label2id) + 1
        label_id = label2id[label]
        colors.append(1. / label_id)
        sizes.append(1500. * G.node[node]['size'])

    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(20, 20)

    nx.draw_networkx(G, cmap=plt.get_cmap('gist_rainbow'),
                     pos=nx.spring_layout(G, k=0.75),
                     node_color=colors,
                     font_color='black',
                     font_size=16,
                     font_weight='bold',
                     alpha=0.75,
                     node_size=sizes,
                     edge_color='gray')

    if show:
        plt.show()
    if save_fpath != "":
        plt.savefig(save_fpath)

    fig.clf()


def get_target_words(language: str) -> List:
    """ Takes as input a two symbol language code e.g. 'de' and returns all 
    words from the evaluation datasets for this language """

    words = set()

    for pairs_fpath in glob("eval/data/{}*dataset".format(language)):
        df = read_csv(pairs_fpath, sep=";", encoding="utf-8")
        for i, row in df.iterrows():
            words.add(row.word1)
            words.add(row.word2)

    words = sorted(words)
    return words


def get_cluster_lines(G, nodes):
    lines = []
    labels_clusters = sorted(aggregate_clusters(G).items(), key=lambda e: len(e[1]), reverse=True)
    for label, cluster in labels_clusters:
        scored_words = []
        for word in cluster:
            scored_words.append((nodes[word], word))
        keyword = sorted(scored_words, reverse=True)[0][1]

        lines.append("{}\t{}\t{}\t{}\n".format(G.name, label, keyword, ", ".join(cluster)))
    return lines


def create_logger(language: str, path: str, name: str = 'info', level=logging.INFO):
    # Create logger for this language
    logger = logging.getLogger("graphVector {} ({})".format(language, name))
    logger.setLevel(level)

    # Create the logging file handler
    log_path = os.path.join(path, name + ".log")
    fh = logging.FileHandler(log_path)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    return logger


def run(language="ru", eval_vocabulary: bool = False, visualize: bool = True,
        show_plot: bool = False, faiss_gpu: bool = True):
    global logger_info, logger_error

    inventory_path = os.path.join("inventories", language)
    log_dir_path = os.path.join(inventory_path, "logs")

    # Create folder for language
    os.makedirs(log_dir_path, exist_ok=True)

    logger_info = create_logger(language=language, path=log_dir_path)
    logger_error = create_logger(language=language, path=log_dir_path, name='error', level=logging.ERROR)

    # Get w2v models paths
    wv_fpath, wv_pkl_fpath = get_embedding_path(language)

    # ensure the word vectors are saved in the fast to load gensim format
    if not exists(wv_pkl_fpath):
        wv = load_globally(wv_fpath, faiss_gpu)  # loads wv
        save_to_gensim_format(wv, wv_pkl_fpath)
    else:
        wv = load_globally(wv_pkl_fpath, faiss_gpu)

    # Get list of words for language
    if eval_vocabulary:
        voc = get_target_words(language)
    else:
        voc = list(wv.vocab.keys())

    words = {w: None for w in voc}

    print("Language:", language)
    print("Visualize:", visualize)
    print("Vocabulary: {} words".format(len(voc)))

    # Load neighbors for vocabulary (globally)
    global voc_neighbors
    voc_neighbors = get_nns_faiss_batch(voc, batch_size=BATCH_SIZE)

    # Init folder for inventory plots
    if visualize:
        plt_path = os.path.join(inventory_path, "plots")
        os.makedirs(plt_path, exist_ok=True)

    # perform word sense induction
    for topn in (50, 100, 200):

        if visualize:
            plt_topn_path = os.path.join(plt_path, str(topn))
            os.makedirs(plt_topn_path, exist_ok=True)

        if verbose:
            print('{} neighbors'.format(topn))

        logger_info.info("{} neighbors".format(topn))

        inventory_file = "cc.{}.300.vec.gz.top{}.inventory.tsv".format(language, topn)
        output_fpath = os.path.join(inventory_path, inventory_file)

        with codecs.open(output_fpath, "w", "utf-8") as out:
            out.write("word\tcid\tkeyword\tcluster\n")

        for index, word in enumerate(words):

            if index + 1 == LIMIT:
                print("OUT OF LIMIT {}".format(LIMIT))

                logger_info.error("OUT OF LIMIT".format(LIMIT))
                break

            if word in string.punctuation:
                print("Skipping word '{}', because it is a punctuation\n".format(word))
                continue

            if verbose:
                print("{} neighbors, word {} of {}, LIMIT = {}\n".format(topn, index + 1, len(words), LIMIT))

            logger_info.info("{} neighbors, word {} of {}, LIMIT = {}".format(topn, index + 1, len(words), LIMIT))

            try:
                words[word] = wsi(word, neighbors_number=topn)
                if visualize:
                    plt_topn_path_word = os.path.join(plt_topn_path, "{}.pdf".format(word))
                    draw_ego(words[word]["network"], show_plot, plt_topn_path_word)
                lines = get_cluster_lines(words[word]["network"], words[word]["nodes"])
                with codecs.open(output_fpath, "a", "utf-8") as out:
                    for line in lines:
                        out.write(line)

            except KeyboardInterrupt:
                break
            except:
                print("Error:", word)
                print(format_exc())
                logger_error.error("{} neighbors, {}: {}".format(topn, word, format_exc()))


def main():
    parser = argparse.ArgumentParser(description='Graph-Vector Word Sense Induction approach.')
    parser.add_argument("language", help="A code that represents input language, e.g. 'en', 'de' or 'ru'. ")
    parser.add_argument("-eval", help="Use only evaluation vocabulary, not all words.", action="store_true")
    parser.add_argument("-viz", help="Visualize each ego networks.", action="store_true")
    parser.add_argument("-gpu", help="Use GPU for faiss", action="store_true")
    args = parser.parse_args()

    run(language=args.language, eval_vocabulary=args.eval, visualize=args.viz, faiss_gpu=args.gpu)


if __name__ == '__main__':
    main()

import os
import re
import codecs
import logging
import argparse
from time import time
from collections import Counter
from traceback import format_exc
from os.path import exists
from typing import List, Dict, Set
import string

import faiss
import numpy as np
import networkx as nx
from networkx import Graph
from gensim.models import KeyedVectors
from chinese_whispers import chinese_whispers, aggregate_clusters

import matplotlib
import matplotlib.pyplot as plt

from load_fasttext import download_word_embeddings

matplotlib.use("pdf")  # speed up graph plot saving

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class GraphInductor(object):
    def __init__(self, language: str, faiss_gpu: bool, gpu_device: int, batch_size: int,
                 chinese_whispers_n: int, limit: int, visualize: int, show_plot: bool = False):

        self.language = language
        self.faiss_gpu = faiss_gpu
        self.gpu_device = gpu_device
        self.batch_size = batch_size
        self.chinese_whispers_n = chinese_whispers_n
        self.limit = limit
        self.visualize = visualize
        self.show_plot = show_plot

        self.inventory_path = os.path.join("inventories", self.language)
        self.log_dir_path = os.path.join(self.inventory_path, "logs")

        self.logger_info = self._create_logger_()
        self.logger_error = self._create_logger_(name='error', level=logging.ERROR)

        self.wv = None
        self.index_faiss = None
        self.voc = None
        self.voc_neighbors = None
        self.plt_path = None
        self.neighbors_number = None

    @staticmethod
    def _filter_voc_(voc):
        """Removes tokens with dot or digits."""
        re_filter = re.compile('^((?![\d.!?{},:()[\]"|/;_+%#<>№»«…*—$]).)*$')
        return [item for item in voc if (re_filter.search(item) is not None) and (item not in string.punctuation)]

    def _get_embedding_path_(self, language):
        """ Ensures that the word vectors exist by downloading them if needed. """

        dir_path = os.path.join("fasttext_models", language)
        wv_fpath = os.path.join(dir_path, "cc.{}.300.vec.gz".format(language))

        if os.path.exists(wv_fpath):
            self.logger_info.info('Embedding for {} exists'.format(language))
        else:
            self.logger_info.info('Embedding for {} does not exist, loading'.format(language))
            download_word_embeddings(language)

        wv_pkl_fpath = wv_fpath + ".pkl"
        return wv_fpath, wv_pkl_fpath

    @staticmethod
    def _save_to_gensim_format_(wv, output_fpath: str):
        """Saves gensim vectors in the gensim format file."""
        tic = time()
        wv.save(output_fpath)
        print("Saved in {} sec.".format(time() - tic))
        return None

    def _load_vectors_(self, word_vectors_fpath: str):
        """Loads gensim vectors from file."""
        self.logger_info.info("Loading word vectors from:", word_vectors_fpath)
        tic = time()
        if word_vectors_fpath.endswith(".vec.gz"):
            wv = KeyedVectors.load_word2vec_format(word_vectors_fpath, binary=False, unicode_errors="ignore")
        else:
            wv = KeyedVectors.load(word_vectors_fpath)
        self.logger_info.info("Loaded in {} sec.".format(time() - tic))
        wv.init_sims(replace=True)
        return wv

    @staticmethod
    def _prepare_faiss_(wv, use_gpu: bool, gpu_device: int):
        """Creates faiss index with word vectors."""
        if use_gpu:
            res = faiss.StandardGpuResources()  # use a single GPU
            index_flat = faiss.IndexFlatIP(wv.vector_size)  # build a flat (CPU) index
            index_faiss = faiss.index_cpu_to_gpu(res, gpu_device, index_flat)  # make it into a gpu index
            index_faiss.add(wv.vectors_norm)  # add vectors to the index
        else:
            index_faiss = faiss.IndexFlatIP(wv.vector_size)
            index_faiss.add(wv.vectors_norm)
        return index_faiss

    def _get_nns_(self, target: str, neighbors_number: int):
        """
        Gets neighbors for the target word.
        :param target: word to find neighbors
        :param neighbors_number: number of neighbors
        :return: list of target neighbors
        """
        target_neighbors = self.voc_neighbors[target]
        if len(target_neighbors) >= neighbors_number:
            return target_neighbors[:neighbors_number]
        else:
            self.logger_error.error("neighbors_number {} is more than precomputed {}".format(neighbors_number, len(target_neighbors)))
            exit(1)

    def _calculate_nns_(self, neighbors_number: int) -> Dict:
        """
        Calculate neighbors for targets by Faiss.
        :param neighbors_number: number of neighbors
        :return: dict of word -> list of neighbors
        """

        word_neighbors_dict = dict()
        self.logger_info.info("Start Faiss with batches")
        for start in range(0, len(self.voc), self.batch_size):
            end = start + self.batch_size
            self.logger_info.info("batch {} to {} of {}".format(start, end, len(self.voc)))
            batch_dict = self.__calculate_nns_batch__(self.voc[start:end], neighbors_number=neighbors_number)
            word_neighbors_dict = {**word_neighbors_dict, **batch_dict}
        return word_neighbors_dict

    def __calculate_nns_batch__(self, targets: List, neighbors_number: int) -> Dict:
        """
        Calculate nearest neighbors for the list of targets for a batch.
        :param targets: list of target words
        :param neighbors_number: number of neighbors
        :return: dict of word -> list of neighbors
        """

        numpy_vec = np.array([self.wv[target] for target in targets])  # Create array of batch vectors
        D, I = self.index_faiss.search(numpy_vec, neighbors_number + 1)  # Find neighbors

        # Write neighbors into dict
        word_neighbors_dict = dict()
        for word_index, (_D, _I) in enumerate(zip(D, I)):
            word = targets[word_index]
            nns_list = []
            for n, (d, i) in enumerate(zip(_D.ravel(), _I.ravel())):
                if n > 0:
                    nns_list.append((self.wv.index2word[i], d))
            word_neighbors_dict[word] = nns_list
        return word_neighbors_dict

    @staticmethod
    def _in_nns_(nns, word: str) -> bool:
        """Checks if word is in list of tuples nns."""
        for w, s in nns:
            if word.strip().lower() == w.strip().lower():
                return True
        return False

    @staticmethod
    def _get_pair_(first, second) -> tuple:
        pair_lst = sorted([first, second])
        sorted_pair = (pair_lst[0], pair_lst[1])
        return sorted_pair

    def _get_disc_pairs_(self, ego, neighbors_number: int) -> Set:
        pairs = set()
        nns = self._get_nns_(ego, neighbors_number)
        nns_words = [row[0] for row in nns]  # list of neighbors (only words)
        wv_neighbors = np.array([self.wv[nns_word] for nns_word in nns_words])
        wv_ego = np.array(self.wv[ego])
        wv_negative_neighbors = (wv_neighbors - wv_ego) * (-1)  # untop vectors
        D, I = self.index_faiss.search(wv_negative_neighbors, 1 + 1)  # find top neighbor for each difference

        # Write down top-untop pairs
        pairs_list_2 = list()
        for word_index, (_D, _I) in enumerate(zip(D, I)):
            for n, (d, i) in enumerate(zip(_D.ravel(), _I.ravel())):
                if self.wv.index2word[i] != ego:  # faiss find either ego-word or untop we need
                    pairs_list_2.append((nns_words[word_index], self.wv.index2word[i]))
                    break

        # Filter pairs
        for pair in pairs_list_2:
            if self._in_nns_(nns, pair[1]):
                pairs.add(self._get_pair_(pair[0], pair[1]))

        return pairs

    @staticmethod
    def _get_nodes_(pairs: Set) -> Counter:
        nodes = Counter()
        for src, dst in pairs:
            nodes.update([src])
            nodes.update([dst])
        return nodes

    @staticmethod
    def _list2dict_(lst: list) -> Dict:
        return {p[0]: p[1] for p in lst}

    def _wsi_(self, ego: str, neighbors_number: int) -> Dict:
        """
        Gets graph of neighbors for word (ego).
        :param ego: word
        :param neighbors_number: number of neighbors
        :return: dict of network and nodes
        """
        tic = time()
        ego_network = Graph(name=ego)

        pairs = self._get_disc_pairs_(ego, neighbors_number)
        nodes = self._get_nodes_(pairs)

        ego_network.add_nodes_from([(node, {'size': size}) for node, size in nodes.items()])

        for r_node in ego_network:
            related_related_nodes = self._list2dict_(self._get_nns_(r_node, neighbors_number))
            related_related_nodes_ego = sorted(
                [(related_related_nodes[rr_node], rr_node) for rr_node in related_related_nodes if
                 rr_node in ego_network],
                reverse=True)[:neighbors_number]

            related_edges = []
            for w, rr_node in related_related_nodes_ego:
                if self._get_pair_(r_node, rr_node) not in pairs:
                    related_edges.append((r_node, rr_node, {"weight": w}))
            ego_network.add_edges_from(related_edges)

        chinese_whispers(ego_network, weighting="top", iterations=self.chinese_whispers_n)
        self.logger_info.info("{}\t{:f} sec.".format(ego, time() - tic))
        return {"network": ego_network, "nodes": nodes}

    def _draw_ego_(self, graph, save_fpath=""):
        """Creates a plot of the graph."""
        tic = time()

        label2id = {}
        colors = []
        sizes = []
        for node in graph.nodes():
            label = graph.nodes[node]['label']
            if label not in label2id:
                label2id[label] = len(label2id) + 1
            label_id = label2id[label]
            colors.append(1. / label_id)
            sizes.append(1500. * graph.nodes[node]['size'])

        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(20, 20)

        nx.draw_networkx(graph,
                         cmap=plt.get_cmap('gist_rainbow'),
                         pos=nx.spring_layout(graph, k=0.75),
                         node_color=colors,
                         font_color='black',
                         font_size=16,
                         font_weight='bold',
                         alpha=0.75,
                         node_size=sizes,
                         edge_color='gray')

        if self.show_plot:
            plt.show()
        if save_fpath != "":
            plt.savefig(save_fpath)
            self.logger_info.info("Plot saved at {}".format(save_fpath))

        fig.clf()

        self.logger_info.info("Created graph plot for: {} sec.".format(time() - tic))
        return None

    @staticmethod
    def _get_cluster_lines_(graph, nodes):
        lines = []
        labels_clusters = sorted(aggregate_clusters(graph).items(), key=lambda e: len(e[1]), reverse=True)
        for label, cluster in labels_clusters:
            scored_words = []
            for word in cluster:
                scored_words.append((nodes[word], word))
            keyword = sorted(scored_words, reverse=True)[0][1]

            lines.append("{}\t{}\t{}\t{}\n".format(graph.name, label, keyword, ", ".join(cluster)))
        return lines

    def _create_logger_(self, name: str = 'info', level=logging.INFO):
        logger = logging.getLogger("graphVector {} ({})".format(self.language, name))
        logger.setLevel(level)

        os.makedirs(self.log_dir_path, exist_ok=True)
        log_path = os.path.join(self.log_dir_path, name + ".log")
        fh = logging.FileHandler(log_path)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        logger.addHandler(fh)
        return logger

    def prepare_vocabulary(self, neighbors_number: int, filter_voc: bool):

        wv_fpath, wv_pkl_fpath = self._get_embedding_path_(self.language)
        self.neighbors_number = neighbors_number

        # ensure the word vectors are saved in the fast to load gensim format
        if not exists(wv_pkl_fpath):
            self.wv = self._load_vectors_(wv_fpath)
            self._save_to_gensim_format_(self.wv, wv_pkl_fpath)
        else:
            self.wv = self._load_vectors_(wv_pkl_fpath)

        self.index_faiss = self._prepare_faiss_(self.wv, self.faiss_gpu, self.gpu_device)
        self.voc = list(self.wv.vocab.keys())

        self.logger_info.info("Language:", self.language)
        self.logger_info.info("Visualize:", self.visualize)
        self.logger_info.info("Vocabulary: {} words".format(len(self.voc)))

        # Load neighbors for vocabulary
        self.voc_neighbors = self._calculate_nns_(neighbors_number=neighbors_number)

        # TODO: change visualize from int to bool
        # Init folder for inventory plots
        if self.visualize:
            self.plt_path = os.path.join("plots", self.language)
            os.makedirs(self.plt_path, exist_ok=True)

        # filter tokens by dots, digits and limit
        if filter_voc:
            self.logger_info.info("Filtering vocabulary...")
            self.voc = self._filter_voc_(self.voc)

        if self.limit < len(self.voc):
            self.voc = self.voc[:self.limit]

        self.logger_info.info("Vocabulary preparation is complete")
        return None

    def run_and_save(self, top_n: int):
        """Performs word sense induction and saves results to a file."""

        words = {w: None for w in self.voc}

        if self.visualize:
            plt_topn_path = os.path.join(self.plt_path, str(top_n))
            os.makedirs(plt_topn_path, exist_ok=True)

        self.logger_info.info("{} neighbors".format(top_n))

        inventory_file = "cc.{}.300.vec.gz.top{}.inventory.tsv".format(self.language, top_n)
        output_fpath = os.path.join(self.inventory_path, inventory_file)

        with codecs.open(output_fpath, "w", "utf-8") as out:
            out.write("word\tcid\tkeyword\tcluster\n")

        for index, word in enumerate(words):

            self.logger_info.info("{} neighbors, word {} of {}".format(top_n, index + 1, len(words)))

            if self.visualize:
                plt_topn_path_word = os.path.join(plt_topn_path, "{}.pdf".format(word))
                if os.path.exists(plt_topn_path_word):
                    self.logger_info.info("Plot for word {} already exists".format(word))
                    continue

            try:
                words[word] = self._wsi_(word, neighbors_number=top_n)
                if self.visualize:
                    self._draw_ego_(words[word]["network"], plt_topn_path_word)
                lines = self._get_cluster_lines_(words[word]["network"], words[word]["nodes"])
                with codecs.open(output_fpath, "a", "utf-8") as out:
                    for line in lines:
                        out.write(line)

            except KeyboardInterrupt:
                break
            except:
                print("Error:", word)
                print(format_exc())
                self.logger_error.error("{} neighbors, {}: {}".format(top_n, word, format_exc()))
        return None


def main():
    parser = argparse.ArgumentParser(description='Graph-Vector Word Sense Induction approach.')
    parser.add_argument("language", help="A code that represents input language, e.g. 'en', 'de' or 'ru'. ")
    parser.add_argument("-viz", help="Visualize each ego networks.", action="store_true")
    parser.add_argument("-gpu", help="Use GPU for faiss", action="store_true")
    parser.add_argument("-filter_voc", help="Filter vocabulary by digits and punctuation", action="store_true")
    parser.add_argument("-gpu_device", help="Which GPU to use", type=int, default=0)
    parser.add_argument("-top_n", help="Number of neighbors", type=int, default=200)
    parser.add_argument("-batch_size", help="How many objects put in faiss per time", type=int, default=2000)
    parser.add_argument("-limit", help="Inventory size", type=int, default=100000)
    parser.add_argument("-cw", help="Number of Chinese Whispers iterations", type=int, default=20)

    args = parser.parse_args()
    graph_inductor = GraphInductor(language=args.language,
                                   faiss_gpu=args.gpu,
                                   gpu_device=args.gpu_device,
                                   chinese_whispers_n=args.cw,
                                   batch_size=args.batch_size,
                                   limit=args.limit,
                                   visualize=args.viz)
    graph_inductor.prepare_vocabulary(args.top_n, args.filter_voc)
    graph_inductor.run_and_save(args.top_n)


if __name__ == '__main__':
    main()

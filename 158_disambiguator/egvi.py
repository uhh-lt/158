from disambiguator import WSD
from pandas import read_csv
import argparse


def evaluate(wsd_model, dataset_fpath, max_context_words):
    """ Evaluates the model using the global variable wsd_model """

    output_fpath = dataset_fpath + ".filter{}.pred.csv".format(max_context_words)
    df = read_csv(dataset_fpath, sep="\t", encoding="utf-8")

    for i, row in df.iterrows():
        sense_id, _ = wsd_model.get_best_sense_id(row.context, row.word, max_context_words)
        df.loc[i, "predict_sense_id"] = sense_id

    df.to_csv(output_fpath, sep="\t", encoding="utf-8", index=False)
    print("Output:", output_fpath)

    return output_fpath


def main():
    parser = argparse.ArgumentParser(description='Sensegram egvi.')
    parser.add_argument("inventory", help="Path of the inventory file.")
    parser.add_argument("-fpath", help="Path of the file to evaluate.", required=True)
    parser.add_argument("-lang", help="Language of the stopwords.", required=True)
    parser.add_argument("-window", help="Context window size.", type=int, required=True)
    args = parser.parse_args()

    wsd_model = WSD(args.inventory, language=args.lang, verbose=True, skip_unknown_words=True)
    evaluate(wsd_model, args.fpath, args.window)


if __name__ == '__main__':
        main()
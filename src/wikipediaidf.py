from argparse import ArgumentParser
from collections import Counter
import logging
import json
import re
import sys
import math
import unicodecsv as csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from bz2 import BZ2File
from multiprocessing import Pool

DROP_TOKEN_RE = re.compile("^\W*$")


def filter_tokens(tokens):
	for t in tokens:
		if not DROP_TOKEN_RE.match(t):
			yield t.lower()


def stem(tokens):
	global stemmer
	stems = []
	token_to_stem_mapping = dict()

	for t in tokens:
		s = stemmer.stem(t)
		stems.append(s)
		if s not in token_to_stem_mapping:
			token_to_stem_mapping[s] = Counter()
		token_to_stem_mapping[s][t] += 1

	return set(stems), token_to_stem_mapping


def get_file_reader(filename):
	if filename.endswith(".bz2"):
		return BZ2File(filename)
	else:
		return open(filename)


def get_lines(input_files):
	for filename in input_files:
		with get_file_reader(filename) as f:
			for line in f:
				yield line


def process_line(line):
	global stemmer
	article_json = json.loads(line)
	tokens = set(filter_tokens(word_tokenize(article_json["text"])))
	stems, token_to_stem_mapping = stem(tokens) if stemmer else None, None
	return tokens, stems, token_to_stem_mapping


def main():
	global stemmer
	parser = ArgumentParser()
	parser.add_argument("-i", "--input", required=True, nargs='+', action="store", help="Input JSON files")
	parser.add_argument("-s", "--stem", metavar="LANG", choices=SnowballStemmer.languages, help="Also produce list of stem words")
	parser.add_argument("-o", "--output", metavar="OUT_BASE", required=True, help="Output CSV files base")
	parser.add_argument("-l", "--limit", metavar="LIMIT", type=int, help="Stop after reading LIMIT articles.")
	parser.add_argument("-c", "--cpus", default=1, type=int, help="Number of CPUs to employ.")
	args = parser.parse_args()

	nltk.download("punkt")

	stemmer = SnowballStemmer(args.stem) if args.stem else None

	tokens_c = Counter()
	stems_c = Counter()
	token_to_stem_mapping = dict()
	articles = 0
	pool = Pool(processes=args.cpus)

	for tokens, stems, t_to_s_mapping in pool.imap_unordered(process_line, get_lines(args.input)):
		tokens_c.update(tokens)
		if args.stem:
			stems_c.update(stems)
			for token in t_to_s_mapping:
				if token not in token_to_stem_mapping:
					token_to_stem_mapping[token] = Counter()
				token_to_stem_mapping[token].update(t_to_s_mapping[token])

		articles += 1
		if not (articles % 100):
			logging.info("Done %d articles.", articles)
		if articles == args.limit:
			break
	pool.terminate()

	with open("{}_{}".format(args.output, "terms.csv"), "w") as o:
		w = csv.writer(o, encoding='utf-8')
		w.writerow(("token", "frequency", "total", "idf"))
		for token, freq in tokens_c.most_common():
			w.writerow([token, freq, articles, math.log(float(articles) / freq)])

	if args.stem:
		with open("{}_{}".format(args.output, "stems.csv"), "w") as o:
			w = csv.writer(o, encoding='utf-8')
			w.writerow(("stem", "frequency", "total", "idf", "most_freq_term"))
			for s, freq in stems_c.most_common():
				w.writerow([s, freq, articles, math.log(articles / (1.0 + freq)), token_to_stem_mapping[s].most_common(1)[0][0]])


if __name__ == "__main__":
	logging.basicConfig(level=logging.DEBUG)
	sys.exit(main())

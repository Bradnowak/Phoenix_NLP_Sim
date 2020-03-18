import os
import docx2txt as docx
import numpy as np
import itertools
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.models import LsiModel
from gensim.models import TfidfModel
from gensim.similarities import MatrixSimilarity


class Extractor:
    def __init__(self, os_dir):
        self.os_dir = os_dir  # Directory to search for documents
        self.dir_list = []  # Search os_dir for document names
        self.stop_words = stopwords.words('english')  # Setup StopWords for elimination of common English words
        self.f_list = []  # Holder for list of file name
        self.data = []  # Holder for files that were actually processed
        self.corpus = []  # Contains all files
        self.tfidf = []
        self.sim_matrix = {}
        self.ind_word_scores = {}

    def gen_f_list(self):
        self.dir_list = os.walk(self.os_dir)
        for i, doc in enumerate(self.dir_list):
            for x, y in enumerate(doc):
                self.f_list = doc[x]

    @staticmethod
    def remove_exemptions(array):
        exempt_strings = []
        filepath = 'exemptions.txt'
        with open(filepath) as fp:
            line = fp.readline()
            while line:
                exempt_strings.append(line.strip('\n'))
                line = fp.readline()
        exempt_strings = np.array(exempt_strings)
        array = np.setdiff1d(array, exempt_strings)
        return array

    # Remove newline characters, tokenize, and remove common english words and non alphabetical terms
    def preprocess(self, text):
        text = text.lower()
        text = text.strip('\n')
        text = text.strip('\t')
        doc = word_tokenize(text)
        doc = [word for word in doc if word not in self.stop_words]
        doc = [word for word in doc if word.isalpha()]
        return doc

    # Return top-n vector of similarities after sorting for highest similarity of a given index (document)
    @staticmethod
    def most_similar(i, data, X_sims, topn=None):
        r = np.argsort(X_sims[i])[::-1]
        scores = np.take(X_sims, r)
        scores = np.sort(scores)[::-1]
        if r is None:
            return data[r], scores, r
        else:
            return data[r[:topn]], scores[:topn], r[:topn]

    @staticmethod
    def convert_pdf(fname, pages=None):
        if not pages:
            page_nums = set()
        else:
            page_nums = set(pages)

        output = StringIO()
        manager = PDFResourceManager()
        converter = TextConverter(manager, output, laparams=LAParams())
        interpreter = PDFPageInterpreter(manager, converter)

        infile = open(fname, 'rb')
        for page in PDFPage.get_pages(infile, page_nums):
            interpreter.process_page(page)
        infile.close()
        converter.close()
        pdf_text = output.getvalue()
        output.close()
        return pdf_text

    # Iterate through all files in file list and perform text extraction/pre-processing
    def process_docs(self):
        for name in self.f_list:
            longName = (self.os_dir + name)

            if ".pdf" in name:
                if PDFPage.get_pages(longName, check_extractable=False):
                    try:
                        text = self.convert_pdf(longName)
                        text = self.preprocess(text)
                        self.corpus.append(text)
                        self.data.append(name)
                    except:
                        print("Unable to parse PDF file: " + longName)

            if ".docx" in name:
                text = docx.process(longName)
                text = self.preprocess(text)
                self.corpus.append(text)
                self.data.append(name)

            elif name.split('.')[-1] == 'doc':
                print(
                    "Detected .doc file: " + name +
                    ". Please convert to .docx or .pdf if you want this file to be included.")

        trigrams = Phrases(self.corpus, min_count=1, threshold=2, delimiter=b' ')
        trigram_phraser = Phraser(trigrams)

        trigram_token = []
        for i in self.corpus:
            trigram_token.append(trigram_phraser[i])

        self.corpus = trigram_token
        for x, arr in enumerate(self.corpus):
            self.corpus[x] = np.array(self.corpus[x])
            self.corpus[x] = self.remove_exemptions(self.corpus[x])
            self.corpus[x] = self.corpus[x].tolist()

    # This function utilizes the gensim package for NLP.
    def find_similarity_scores(self, topics):
        # Create similarities container
        similarities = {'Resumes': {}}
        # Gensim requires a corpora data structure for transformations and analysis
        dictionary = corpora.Dictionary(self.corpus)

        # Convert text to BoW.  It already is but lets be sure.
        corpus_gensim = [dictionary.doc2bow(doc) for doc in self.corpus]

        # Term Frequency-Inverse Document Frequency (TF-IDF) transformation sets weights small
        # when they appear more often in the text.
        self.tfidf = TfidfModel(corpus_gensim)
        print(self.tfidf)
        self.tfidf = self.tfidf[corpus_gensim]
        print(self.tfidf)
        # Find similarity via vector-space pair-wise cosine angle absolute value via Latent Semantic Indexing (LSI)
        # https://en.wikipedia.org/wiki/Latent_semantic_analysis#Latent_semantic_indexing
        lsi = LsiModel(self.tfidf, id2word=dictionary, num_topics=topics)
        lsi_index = MatrixSimilarity(lsi[self.tfidf])
        similarities['Resumes']["LSI_Similarity"] = np.array(
            [lsi_index[lsi[self.tfidf[i]]] for i in range(len(self.corpus))])


        for doc in self.tfidf:
            for word_id, value in doc:
                word = dictionary.get(word_id)
                self.ind_word_scores[word] = value

        # Convert to numpy arrays
        self.f_list = np.array(self.f_list)
        self.data = np.array(self.data)

        # Return results to object
        self.sim_matrix = similarities

    # Query the model and print the n most similar documents.
    def query_doc(self, doc, n):
        print("Most similar documents for " + doc + ':\n')
        file_num = np.where(self.data == doc)[0][0]
        n_list, scores, indices = (
            self.most_similar(file_num, self.data, self.sim_matrix['Resumes']['LSI_Similarity'], n))

        for document, score, index in zip(n_list, scores, indices):
            print("\n" + document)
            print("Similarity Score: " + str(score))
            print("Shared Words: " + str(np.intersect1d(self.corpus[index], self.corpus[file_num])))

    #  Find the n most similar documents in the corpus
    #  Find the most dissimilar document in the corpus and trims it recursively until only n documents are left.
    def find_most_similar_in_corpus(self, n):
        sims_copy = np.array(self.sim_matrix['Resumes']['LSI_Similarity'])
        data_copy = self.data
        while sims_copy.shape[0] > n:
            sims_sums = np.sum(sims_copy, axis=1)
            trim_index = np.where(sims_sums == sims_sums.min())[0][0]
            sims_copy = np.delete(sims_copy, trim_index, 0)
            sims_copy = np.delete(sims_copy, trim_index, 1)
            data_copy = np.delete(data_copy, trim_index)
        print(data_copy)

    def merge_common(self, topn):
        words = []
        word_counts = []
        word_counts_min = []
        for x in topn:
            word_index = np.where(self.data == x)[0][0]
            words.append(self.corpus[word_index])
        words = list(itertools.chain.from_iterable(words))
        for word in words:
            num = words.count(word)
            word_counts.append((word, num))
        word_counts = np.array(word_counts, dtype=[('w', 'U80'), ('n', 'int')])
        word_counts = np.unique(word_counts)
        word_counts[::-1].sort(order='n')  # Reverse order sort by number of word occurrences
        print("\nMost common words:")
        print(word_counts[:15])
        for word in word_counts:
            if word[1] > 1:
                word_counts_min.append((word[0], word[1] * self.ind_word_scores[word[0]]))
        word_counts_min = np.array(word_counts_min, dtype=[('w', 'U25'), ('n', 'float64')])
        word_counts_min[::-1].sort(order='n')  # Reverse order sort by number of word occurrences
        print("\nMost significant words:")
        print(word_counts_min[:10])

    def find_most_similar_in_corpus_all(self, n):
        data_copy = self.data
        data_copy2 = self.data
        count = 0
        mask = np.full(len(self.data), False, dtype=bool)

        while data_copy.shape[0] > n:
            sims_copy = np.array(self.sim_matrix['Resumes']['LSI_Similarity'])
            remove_array = np.where(mask)
            sims_copy = np.delete(sims_copy, remove_array, 0)
            sims_copy = np.delete(sims_copy, remove_array, 1)
            while data_copy.shape[0] > n:
                sims_sums = np.sum(sims_copy, axis=1)
                trim_index = np.where(sims_sums == sims_sums.min())[0][0]
                sims_copy = np.delete(sims_copy, trim_index, 0)
                sims_copy = np.delete(sims_copy, trim_index, 1)
                data_copy = np.delete(data_copy, trim_index)
            print("\n" + "-" * 20 + "Top " + str(count + 1) + " to " + str(count + n) + " most similar documents" + "-" * 20)
            count += n
            print(data_copy)
            print("\n" + "-" * 20 + "Common Words" + "-" * 20)
            self.merge_common(data_copy)
            data_copy2 = np.setdiff1d(data_copy2, data_copy)
            data_copy = data_copy2
            comp = np.isin(self.data, data_copy, invert=True)
            mask = np.logical_or(mask, comp)
        print("\n" + "-" * 20 + "Remaining documents" + "-" * 20)
        print(data_copy)
        self.merge_common(data_copy)


    #  Find the n most documents in the corpus
    #  Flattens array of similarity scores, then sorts them smallest to largest.  Iterate through similarity scores.
    #  For each document that doesn't exist in a pair we create an array representing that document.  For each score
    #  we also check the arrays to see if one of the documents making up the array is in the pair making up the score.
    #  If it is, then it is also added to the array.  The first array to reach n documents is the most similar
    #  group of docs.
    def find_most_similar_in_corpus_2(self, n):
        sims_copy = np.array(self.sim_matrix['Resumes']['LSI_Similarity'])
        sims_copy = np.tril(sims_copy, -1)
        sims_copy = np.ndarray.flatten(sims_copy)
        print(sims_copy)
        sims_copy = np.where(sims_copy[0] != 0)
        print(type(sims_copy))
        sims_copy = sims_copy[0].sort()
        for i in sims_copy[1]:
            position = np.where(self.sim_matrix == i)
            doc1 = position[0]
            doc2 = position[1]
            print(doc1, doc2)

    def analyze(self, n, topics):
        self.gen_f_list()
        self.process_docs()
        self.find_similarity_scores(topics)
        self.find_most_similar_in_corpus_all(n)

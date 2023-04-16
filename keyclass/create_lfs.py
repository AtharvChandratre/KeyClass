import nltk
from sklearn.semi_supervised import LabelPropagation
import torch
import models
nltk.download('stopwords')
from nltk import download, pos_tag, corpus
from scipy.spatial import distance
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import pandas as pd


def get_vocabulary(text_corpus, max_df=1.0, min_df=0.01, ngram_range=(1, 1)):

    strip_accents = 'unicode'
    stopwords = corpus.stopwords.words('english')
    vectorizer = CountVectorizer()
    vectorizer.max_df = max_df
    vectorizer.min_df = min_df
    vectorizer.strip_accents = strip_accents
    vectorizer.stop_words = stopwords
    vectorizer.ngram_range = ngram_range

    transformed = vectorizer.fit_transform(text_corpus)
    word_indicator_matrix = transformed.toarray()

    feature_names_out = vectorizer.get_feature_names_out()
    vocabulary = np.asarray(feature_names_out)

    return word_indicator_matrix, vocabulary


def assign_categories_to_keywords(vocabulary, vocabulary_embeddings, label_embeddings, word_indicator_matrix, cutoff=None, topk=None, min_topk=True):

    cutoffIsNone = cutoff is None
    topkIsNone = topk is None
    assert (cutoffIsNone or topkIsNone)

    metric = 'cosine'

    distances = distance.cdist(vocabulary_embeddings, label_embeddings, metric)
    axis = 1
    dist_to_closest_cat = np.min(distances, axis=axis)
    assigned_category = np.argmin(distances, axis=axis)

    if cutoff is not None:
        mask = (dist_to_closest_cat <= cutoff)
        mask = mask.astype(bool)

    if topk is not None:
        uniques = np.unique(assigned_category)
        len_dist_closest_to_cat = len(dist_to_closest_cat)
        mask = np.zeros(len_dist_closest_to_cat, dtype=bool)

        counts = np.unique(assigned_category, return_counts=True)[1]
        print('Found assigned category counts', counts)
        topk = np.min([topk, np.min(counts)]) if min_topk is True else topk

        for u in uniques:
            u_inds,  = np.where(assigned_category == u)
            u_dists = dist_to_closest_cat[u_inds]
            all_sorted_inds = np.argsort(u_dists)
            sorted_inds = all_sorted_inds[:topk]
            mask[u_inds[sorted_inds]] = 1

    keywords = vocabulary[mask]
    assigned_category = assigned_category[mask]
    maskForWordIndicatorMatrix = np.where(mask)[0]
    word_indicator_matrix = word_indicator_matrix[:, maskForWordIndicatorMatrix]
    return keywords, assigned_category, word_indicator_matrix

def create_label_matrix(word_indicator_matrix, keywords, assigned_category):

    word_indicator_matrix = np.where(word_indicator_matrix == 0, -1, 0)
    len_assigned_category = len(assigned_category)
    for i in range(len_assigned_category):
        range_word_indicator_matrix = word_indicator_matrix[:, i]
        ith_assigned_category = assigned_category[i]
        word_indicator_matrix[:,i] = np.where(range_word_indicator_matrix != -1, ith_assigned_category, -1)

    return pd.DataFrame(word_indicator_matrix, columns=keywords)


class CreateLabellingFunctions:
    def __init__(self, base_encoder='paraphrase-mpnet-base-v2', device: torch.device = torch.device("cuda"), label_model: str = 'data_programming'):

        self.device = device
        self.encoder = models.Encoder(model_name=base_encoder, device=device)

        self.label_matrix = None
        self.keywords = None
        self.word_indicator_matrix = None
        self.vocabulary = None
        self.vocabulary_embeddings = None
        self.assigned_category = None
        self.label_model_name = label_model

    def get_labels(self,
                   text_corpus,
                   label_names,
                   min_df,
                   ngram_range,
                   topk,
                   y_train,
                   label_model_lr,
                   label_model_n_epochs,
                   verbose=True,
                   n_classes=2):

        self.label_embeddings = self.encoder.encode(sentences=label_names)

        ## get vocab according to n-grams
        self.matrix_and_vocabulary = get_vocabulary(\
            text_corpus=text_corpus,
            max_df=1.0,
            min_df=min_df,
            ngram_range=ngram_range)
        self.word_indicator_matrix = self.matrix_and_vocabulary[0]
        self.vocabulary = self.matrix_and_vocabulary[1]

        self.vocabulary_embeddings = self.encoder.encode(sentences=self.vocabulary)

        self.assigned_categories_to_keywords = assign_categories_to_keywords(\
            vocabulary=self.vocabulary,
            vocabulary_embeddings=self.vocabulary_embeddings,
            label_embeddings=self.label_embeddings,
            word_indicator_matrix=self.word_indicator_matrix,
            topk=topk)
        self.keywords = self.assigned_categories_to_keywords[0]
        self.assigned_category = self.assigned_categories_to_keywords[1]
        self.word_indicator_matrix = self.assigned_categories_to_keywords[2]

        if verbose:
            shape = self.word_indicator_matrix.shape
            vocab = self.vocabulary
            print('labeler.vocabulary:\n', len(vocab))
            print('labeler.word_indicator_matrix.shape', shape)

            keywords = self.keywords
            print('Len keywords', len(keywords))

            assigned_category = self.assigned_category
            print('assigned_category: Unique and Counts',
                  np.unique(assigned_category, return_counts=True))

            for u in range(len(label_names)):
                inds = np.where(self.assigned_category == u)[0]
                inds_keywords = self.keywords[inds]
                u_label = label_names[u]
                print(u_label, inds_keywords)

        return models.LabelModelWrapper(\
            label_matrix=create_label_matrix(\
            word_indicator_matrix=self.word_indicator_matrix,
            keywords=self.keywords,
            assigned_category=self.assigned_category),
            n_classes=n_classes,
            y_train=y_train,
            device=self.device,
            model_name=self.label_model_name).train_label_model(\
            lr=label_model_lr,
            n_epochs=label_model_n_epochs,
            cuda=True if torch.cuda.is_available() else False).predict_proba().values
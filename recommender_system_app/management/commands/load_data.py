from django.core.management.base import BaseCommand
import os
import optparse
import numpy as np
import pandas as pd
import math
import json
import copy
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
tknzr = WordPunctTokenizer()
stoplist = stopwords.words('english');
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer();
from sklearn.feature_extraction.text import TfidfVectorizer
from recommender_system_app.models import MovieData
from django.core.cache import cache


#python manage.py load_data --input=plots.csv --nmaxwords=30000 --umatrixfile=umatrix.csv

class Command(BaseCommand):

    def add_arguments(self, parser):
        # plots.csv file- Contains movie title and plot description
        parser.add_argument('--input', dest='input',
                             type=str, action='store',
                             help='Input plots file');

        # Limits the tfidfvectorizer table with nmaxwords features
        parser.add_argument('--nmaxwords', dest='nmaxwords',
                             type=int, action='store',
                             help='nmaxwords');

        #Utility matrix
        parser.add_argument('--umatrixfile', dest='umatrixfile',
                             type=str, action='store',
                             help='Utility matrix');

    #Remove stop words and apply stemming
    def preprocessTfidf(self, texts, stoplist=[], stem = False):
        newtexts = [];
        for i in range(len(texts)):
            text = texts[i];
            if(stem == False):
                tmp = [w for w in tknzr.tokenize(text) if w not in stoplist]
            else:
                tmp = [stemmer.stem(w) for w in [w for w in tknzr.tokenize(text) if w not in stoplist]]
            newtexts.append(' '.join(tmp));
        return newtexts;

    def handle(self, *args, **options):
        input_file = options['input']

        df = pd.read_csv(input_file)
        tot_textplots = df['plot'].tolist();
        tot_titles = df['title'].tolist();
        nmaxwords = options['nmaxwords'];
        vectorizer = TfidfVectorizer(min_df = 0, max_features = nmaxwords);
        processed_plots = self.preprocessTfidf(tot_textplots, stoplist, True);
        vec_tfidf = vectorizer.fit_transform(processed_plots);
        ndims = len(vectorizer.get_feature_names())
        nmovies = len(tot_titles[:])

        #Clear all data
        MovieData.objects.all().delete();

        #vec_ifidf as an array
        matr = np.empty([0, ndims]);
        titles = [];
        cnt = 0;

        #Fill in records
        for m in range(nmovies):
            moviedata = MovieData();
            moviedata.title = tot_titles[m];
            moviedata.description = tot_textplots[m];
            moviedata.ndim = ndims;
            #json dumps converts the json object into a json formatted string.
            #Holds the tfidf representation of this movie as a string
            tfidf_movie = vec_tfidf[m].toarray()[0].tolist();
            moviedata.array = json.dumps(tfidf_movie); #A string representing the list with the tfidf representation of
                                                       #this movie.
            moviedata.save();

            matr = np.vstack([matr, tfidf_movie]);
            titles.append(moviedata.title);
            cnt += 1;
        #matr can also be calculated: matr = vec_tfidf.toarray()
        print(matr.shape)
        #cached
        cache.set('data', matr);   #tfidf representation of the movies as a dense array
        cache.set('titles', titles); #List of titles
        titles = cache.get('titles');

        print('len:', len(titles));
        cache.set('model', vectorizer);  #Tfidf model


        #Load the utility matrix with the movie ratings.
        umatrixfile = options['umatrixfile'];
        df_umatrix = pd.read_csv(umatrixfile);
        Umatrix = df_umatrix.values[:, 1:];   #Utility matrix withouot the user id column
        print('umatrix: ', Umatrix.shape)
        cache.set('umatrix', Umatrix); #Utility matrix that is used by the recommender systems

        #Load recommender systems...
        cf_itembased = CF_itembased(Umatrix);
        cache.set('cf_itembased', cf_itembased);
        llr = LogLikelihood(Umatrix, titles);
        cache.set('loglikelihood', llr);

from scipy.stats import pearsonr
from scipy.spatial.distance import cosine

#Similarity measure
def sim(x, y, metric= 'cos'):
    if metric == 'cos':
        return 1.0 - cosine(x, y);
    else:
        return pearsonr(x, y)[0];


class CF_itembased:
    def __init__(self, data):
        #Calculated the similarity matrix between all items.
        #The items correspond to the columns in matrix data.
        n_items = len(data[0]);
        self.data = data;
        self.sim_matrix = np.zeros((n_items, n_items));
        #print(data)
        for i in range(n_items):
            for j in range(n_items):
                if j >=i:
                    #print(data[:, i])
                    #print(data[:, j])
                    #print(cosine(data[:, i], data[:, j]))
                    self.sim_matrix[i, j] = sim(data[:, i], data[:, j]);
                else:
                    self.sim_matrix[i, j] = self.sim_matrix[j, i];


    #Gets the K most similar items to r using the items in u_vec that user has already rated.
    def get_ksim_items_per_user(self, r, K, u_vec):
        items = np.argsort(self.sim_matrix[r])[::-1];
        items = items[items!=r];
        cnt = 0;
        neigh_items = [];
        for i in items:
            if u_vec[i] > 0 and cnt < K:
                neigh_items.append(i);
                cnt += 1;
            elif(cnt == K):
                break;
        #Returns the a list with indexes of items.
        return neigh_items;

    #Calculates the rating of item r for user u, using the ratings that user u has given to the most similar
    #items to item r.
    def calc_rating(self, r, u_vec, neigh_items):
        rating = 0.0;
        den = 0.0
        #Weighted sum
        for i in neigh_items:
            rating += self.sim_matrix[r, i]*u_vec[i]
            den += abs(self.sim_matrix[r, i]);
        if(den > 0):
            rating = np.round(rating/den, 0);
        else:
            #Returns an average of the ratings that item r has received by all users.
            rating = np.round(self.data[:, r][self.data[:, r] > 0].mean(), 0);
        return rating;

    #Calculates the ratings for the entries in u_vec that are not rated yet (r) using the K most similar neighbors.
    def calc_ratings(self, u_vec, K, indxs = False):
        u_rec = copy.copy(u_vec)
        for r in range(len(u_vec)):
            if(u_vec[r] == 0):
                neigh_items = self.get_ksim_items_per_user(r, K, u_vec);
                #calc predicted rating
                u_rec[r] = self.calc_rating(r, u_vec, neigh_items);

        if indxs:
            #Return only the rated movies from bigger to smaller
            seenindxs = [indx for indx in range(len(u_vec)) if u_vec[indx] > 0];
            u_rec[seenindxs] = -1;
            recsvec = np.argsort(u_rec)[::-1][np.sort(u_rec)[::-1] > -1];

            return recsvec;

        return u_rec;

class LogLikelihood:
    def __init__(self, Umatrix, movies_list, like_threshold = 3):
        self.movies_list = movies_list;
        self.n_users = len(Umatrix);
        self.Umatrix = Umatrix;
        self.like_threshold = like_threshold;
        self.like_range = range(self.like_threshold + 1, 5 + 1);
        self.dislike_range = range(1, self.like_threshold + 1);
        self.loglikelihood_ratio();

    #Calculated the confusion matrix for item a, b   -- >   a and b, a and not b, not a and b, not a and not a.
    def calc_k(self, a, b):
        tmpk = [[0 for j in range(2)] for i in range(2)]
        for ratings in self.Umatrix:
            if ratings[a] in self.like_range and ratings[b] in self.like_range:
                tmpk[0][0] += 1;
            if ratings[a] in self.like_range and ratings[b] in self.dislike_range:
                tmpk[0][1] += 1;
            if ratings[a] in self.dislike_range and ratings[b] in self.like_range:
                tmpk[1][0] += 1;
            if ratings[a] in self.dislike_range and ratings[b] in self.dislike_range:
                tmpk[1][1] +=1;
        return tmpk;

    #Calculate the LLR = 2*N*I(a, b) = 2*N*(-H(a, b) + H(a) + H(b)) for each pair of elements
    #The k_matrix is the confusion matrix for item a, b;
    def calc_llr(self, k_matrix):
        Hcols = Hrows = Htot = 0.0;
        if(sum(k_matrix[0]) + sum(k_matrix[1]) == 0):
            return 0.0;
        invN = 1.0/(sum(k_matrix[0]) + sum(k_matrix[1]));
        for i in range(0, 2):
            if((k_matrix[0][i] + k_matrix[1][i]!=0.0)):
                Hcols += invN*(k_matrix[0][i] + k_matrix[1][i])*math.log((k_matrix[0][i]+ k_matrix[1][i])*invN);
            if((k_matrix[i][0] + k_matrix[i][1]!=0.0)):
                Hrows += invN * (k_matrix[i][0] + k_matrix[i][1]) * math.log((k_matrix[i][0] + k_matrix[i][1]) * invN);
            for j in range(0, 2):
                if(k_matrix[i][j] != 0.0):
                    Htot += invN*k_matrix[i][j]*math.log(invN*k_matrix[i][j])
        llr = 2.0*(Htot - Hcols - Hrows)/invN;
        #print("Llr: ", llr)
        return llr;

    #Calculate the LLR matrix.
    def loglikelihood_ratio(self):
        n_items = len(self.movies_list);
        self.items_llr = pd.DataFrame(np.zeros((n_items, n_items))).astype(float);
        for i in range(n_items):
            if i % 10 == 0:
                print("Processed ", i, " items out of ", n_items);
            for j in range(n_items):
                if(j>=i):
                    tmpk = self.calc_k(i, j);
                    self.items_llr.loc[i, j] = self.calc_llr(tmpk);
                else:
                    self.items_llr.loc[i, j] = self.items_llr.iat[j, i];
        #print(self.items_llr);

    #Return the recommended items from bigger to smaller
    def get_rec_items(self, u_vec, indxs = False):
        #matrix items_llr elements are always positive.
        #u_vec is has also positive elements. So items_weight has also positive elements.
        items_weight = np.dot(u_vec, self.items_llr);
        sorted_weight = np.argsort(items_weight);
        seenindxs = [indx for indx in range(len(u_vec)) if u_vec[indx] > 0];
        seenmovies = np.array(self.movies_list)[seenindxs]
        #remove seen items
        recitems = np.array(self.movies_list)[sorted_weight];
        recitems = [m for m in recitems if m not in seenmovies];
        if(indxs):
            items_weight[seenindxs]=-1;
            recsvec = np.argsort(items_weight)[::-1][np.sort(items_weight)[::-1]>-1];
            return recsvec;
        return recitems[::-1]
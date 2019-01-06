from django.shortcuts import render
from django.contrib.auth.models import User
from recommender_system_app.models import MovieData,MovieRated,UserProfile
from django.contrib.auth import authenticate, login
from django.contrib.auth import logout
from django.shortcuts import redirect
from django.core.cache import cache
from django.urls import reverse
from ast import literal_eval
import urllib
import copy
import math
import json


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#ñimport nltk
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
tknzr = WordPunctTokenizer()
#nltk.download('stopwords')
stoplist = stopwords.words('english')
from nltk.stem.porter import PorterStemmer
from scipy.stats import beta;
stemmer = PorterStemmer();


from scipy.stats import pearsonr
from scipy.spatial.distance import cosine

umatrixpath = 'umatrix.csv'
nmoviesperquery=5
nminimumrates=5
numrecs=5
recmethod = 'loglikelihood'

# Create your views here.


class Banner():
    def __init__(self, id):
        self.id = id;
        self.clicks = 0;
        self.views = 0;

    def sample(self):
        # Beta prior B(1, 1)
        a = 1 + self.clicks
        b = 1 + self.views - self.clicks
        return np.random.beta(a, b)

    def add_click(self):
        self.clicks += 1

    def add_view(self):
        self.views += 1;

    def get_stats(self):
        print("Banner {:d} Clicks: {:d} Views: {:d} CTR: {:.4f}"
              .format(self.id, self.clicks, self.views, self.clicks/self.views))

'''Perform a movie search given a query. The search is performed used a tfidf model.
If the query is empty we redirect to a home.html page. Else we perform the movie search and return the results
in the query_results.html.
'''


class iterables:
    def __init__(self, titles, indexes):
        self.titles = titles;
        self.indexes = indexes;

    def __str__(self):
        return "Movie zip"

    def __repr__(self):
        return "iterables({}, {})".format(self.titles, self.indexes)

def banner(request, pk):
    context = {}
    if(pk == 1):
        return render(request, 'recommender_system_app/banner_1.html', context)
    elif(pk == 2):
        return render(request, 'recommender_system_app/banner_2.html', context)
    elif(pk == 3):
        return render(request, 'recommender_system_app/banner_3.html', context)

def banner_clicked(request, pk):
    context = {}
    banner_models = cache.get('banner_models')
    banner_models[pk].add_click()
    cache.set('banner_models', banner_models)
    banner_models[pk].get_stats();
    return redirect('https://www.glassmovie.com/')

def process_banner_to_show():
    banner_models = cache.get('banner_models_')
    if(banner_models is None):
        banner_models = {1:Banner(1), 2: Banner(2), 3: Banner(3)}

    max_p = -np.inf
    best_banner = None;
    for key, banner in banner_models.items():
        p = banner.sample();
        if(p > max_p):
            max_p = p;
            best_banner = banner;

    best_banner.add_view();
    cache.set("banner_models_", banner_models);
    return best_banner.id;


def home(request):
    context = {}
    banner_id = process_banner_to_show();
    context["banner_id"] = banner_id;

    print(request.path)

    if request.method == 'POST':
        #The form sends data through a post request. We need to take the data and transform it into a query form
        post_data = request.POST
        data = {}
        data = post_data.get('data', None)
        print("Entering to post ", data)
        if data:
            return redirect('%s?%s' % (reverse('home'),
                                       urllib.parse.urlencode({'q': data})))
    elif request.method == 'GET':
        print("Entering to GET")
        get_data = request.GET
        data = get_data.get('q', None)
        titles = cache.get('titles')
        #If data is already in the cache
        if titles is not None:
            print('loaded from the cache', str(len(titles)))
        else:
            print('load data for the first time...')
            texts = []
            mobjs = MovieData.objects.all()
            ndim = mobjs[0].ndim
            matr = np.empty([1, ndim])
            titles = []
            cnt = 0
            for obj in mobjs[:]:
                texts.append(obj.description)
                newrow = np.array(obj.array)
                if cnt == 0:
                    matr[0] = newrow
                else:
                    matr = np.vstack([matr, newrow])
                titles.append(obj.title)
                cnt += 1
            vectorizer = TfidfVectorizer(min_df=1, max_features=ndim)
            processedtexts = PreprocessTfidf(texts, stoplist, True)
            model = vectorizer.fit(processedtexts)
            cache.set('model', model)
            cache.set('data', matr)
            cache.set('titles', titles)

        Umatrix = cache.get('umatrix')
        if Umatrix is None:
            df_umatrix = pd.read_csv(umatrixpath)
            Umatrix = df_umatrix.values[:, 1:]
            print('umatrix:', Umatrix.shape)
            cache.set('umatrix', Umatrix)
            cf_itembased = CF_itembased(Umatrix)
            cache.set('cf_itembased', cf_itembased)
            cache.set('loglikelihood', LogLikelihood(Umatrix, titles))

        if not data:
            return render(request, 'recommender_system_app/home.html', context)

        # load all movies vectors/titles
        matr = cache.get('data')
        #print('matr', len(matr))
        titles = cache.get('titles')
        #print('ntitles:', len(titles))
        model_tfidf = cache.get('model')

        # find movies similar to the query using a tfidf model
        queryvec = model_tfidf.transform([data.lower().encode('ascii', 'ignore')]).toarray()

        sims = cosine_similarity(queryvec, matr)[0]
        indxs_sims = list(sims.argsort()[::-1])[:nmoviesperquery] #Sort by most similar to least similar
        # print indxs_sims
        titles_query = list(np.array(titles)[indxs_sims][:nmoviesperquery])

        context['movies'] = list(zip(titles_query, indxs_sims))
        context['rates'] = [1, 2, 3, 4, 5]
        return render(request, 'recommender_system_app/query_results.html', context)



def auth(request):
    print('auth--:',request.user.is_authenticated)
    if request.method == 'GET':
        data = request.GET
        auth_method = data.get('auth_method')
        #Render a sign in page
        if auth_method=='sign in':
           return render(request, 'recommender_system_app/signin.html', {})
        #Render a registration page
        else:
            return render(request, 'recommender_system_app/createuser.html', {})

    #Receives the data from the create user page or the log sig in page.
    elif request.method == 'POST':
        post_data = request.POST
        name = post_data.get('name', None)
        pwd = post_data.get('pwd', None)
        pwd1 = post_data.get('pwd1', None)
        print('auth:',request.user.is_authenticated)
        create = post_data.get('create', None)#hidden input
        #If receives data from the create user page
        if name and pwd and create:
           if User.objects.filter(username=name).exists() or pwd!=pwd1:
               return render(request, 'recommender_system_app/userexistsorproblem.html', {})

           #Creates a new user and the corresponding user profile obect
           user = User.objects.create_user(username=name,password=pwd)
           uprofile = UserProfile()
           uprofile.user = user
           uprofile.name = user.username
           uprofile.save(create=True)
           #Authenticates and logins the user immediately
           user = authenticate(username=name, password=pwd)
           login(request, user)
           #Redirects to the home page
           return render(request, 'recommender_system_app/home.html', {})

        #If receives data from the sig in page
        elif name and pwd:
            user = authenticate(username=name, password=pwd)
            if user:
                login(request, user)
                return render(request, 'recommender_system_app/home.html', {})
            else:
                #notfound
                return render(request, 'recommender_system_app/nopersonfound.html', {})


def signout(request):
    logout(request)
    return render(request, 'recommender_system_app/home.html', {})


def rate_movie(request):
    data = request.GET
    rate = data.get("vote")
    print(request.user.is_authenticated)
    #Recover movie names and movie indexes
    movies, moviesindxs = list(zip(*literal_eval(data.get("movies"))))

    #Movie name
    movie = data.get("movie")
    #Movie index in the movie list
    movieindx = int(data.get("movieindx"))

    #Save movie rate. We need to sigin first to rate movies.
    userprofile = None

    print("User logged: {0}".format(request.user))

    print(UserProfile.objects.all())
    if request.user.is_superuser:
        return render(request, 'recommender_system_app/superusersignin.html', {})
    elif request.user.is_authenticated:
        userprofile = UserProfile.objects.get(user=request.user)
    else:
        return render(request, 'recommender_system_app/pleasesignin.html', {})

    #At the beggining there are no objects in the MovieRated table. Only new users will create MovieRated objects.
    if MovieRated.objects.filter(movie=movie).filter(user=userprofile).exists():
        #Update movie rating related to user UserProfile
        mr = MovieRated.objects.get(movie=movie, user=userprofile)
        mr.value = int(rate)
        mr.save()
    else:
        #Create a new movie rating.
        mr = MovieRated()
        mr.user = userprofile
        mr.value = int(rate)
        mr.movie = movie
        mr.movieindx = movieindx
        mr.save()

    #To update its ratedmovies collection
    userprofile.save()
    # get back the remaining movies
    movies = remove_from_list(movies, movie)
    moviesindxs = remove_from_list(moviesindxs, movieindx)
    print(movies)
    context = {}
    context["movies"] = list(zip(movies, moviesindxs))
    context["rates"] = [1, 2, 3, 4, 5]
    return render(request, 'recommender_system_app/query_results.html', context)


def movies_recs(request):
    userprofile = None
    context = {}
    banner_id = process_banner_to_show();
    context["banner_id"] = banner_id;
    print('is super user 2:', request.user.is_superuser, context["banner_id"])
    if request.user.is_superuser:
        print("here1")
        return render(request,
            'recommender_system_app/superusersignin.html', {})
    elif request.user.is_authenticated:
        print("here2")
        userprofile = UserProfile.objects.get(user=request.user)
    else:
        print("here3")
        return render(request, 'recommender_system_app/pleasesignin.html', {})
    print("here4")
    ratedmovies = userprofile.ratedmovies.all()
    print('rated:', ratedmovies, '--', [r.movieindx for r in ratedmovies])

    if len(ratedmovies) < nminimumrates:
        context['nrates'] = len(ratedmovies)
        context['nminimumrates'] = nminimumrates
        return render(request, 'recommender_system_app/underminimum.html', context)
    print("here5")
    u_vec = np.array(json.loads(userprofile.array))

    Umatrix = cache.get('umatrix')
    # print(Umatrix.shape,'--',len(u_vec))
    movieslist = cache.get('titles')
    # recommendation...
    u_rec = None

    if recmethod == 'cf_itembased':
        print(recmethod)
        cf_itembased = cache.get('cf_itembased')
        if cf_itembased == None:
            cf_itembased = CF_itembased(Umatrix)
        u_rec = cf_itembased.calc_ratings(u_vec, numrecs)

    elif recmethod == 'loglikelihood':
        print(recmethod)
        llr = cache.get('loglikelihood')
        if llr == None:
            llr = LogLikelihood(Umatrix, movieslist)
        u_rec = llr.get_rec_items(u_vec, True)

    # save last recs
    userprofile.save(recsvec=u_rec)
    context['recs'] = list(np.array(movieslist)[list(u_rec)][:numrecs])
    print("Context to show")
    print(context)
    return render(request, 'recommender_system_app/recommendations.html', context)


def remove_from_list(list_items,item):
    outlist = []
    for i in list_items:
        if i==item:
            continue
        outlist.append(i)
    return outlist

def PreprocessTfidf(texts,stoplist=[],stem=False):
    newtexts = []
    for text in texts:
        if stem:
           tmp = [w for w in tknzr.tokenize(text) if w not in stoplist]
        else:
           tmp = [stemmer.stem(w) for w in [w for w in tknzr.tokenize(text) if w not in stoplist]]
        newtexts.append(' '.join(tmp))
    return newtexts

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
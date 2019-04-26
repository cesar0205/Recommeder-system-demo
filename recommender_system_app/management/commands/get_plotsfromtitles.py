from django.core.management.base import BaseCommand
import os
import optparse
import numpy as np
import json
import pandas as pd
import requests


#python manage.py get_plotsfromtitles --input=./recommender_system_app/management/commands/utilitymatrix.csv --outputplots=plots.csv --outputumatrix=umatrix.csv
class Command(BaseCommand):

    def add_arguments(self, parser):

        parser.add_argument('--input', dest='umatrixfile',
                             type=str, action='store',
                             help='Input utility matrix');

        parser.add_argument('--outputplots', dest='plotsfile',
                             type=str, action='store',
                             help='output file');

        parser.add_argument('--outputumatrix', dest='umatrixoutfile',
                             type=str, action='store',
                             help='output file');

    '''Gets the the movie's plot. Additionaly creates a new utility matrix that it is equal to the input utility matrix
        when all the movies' titles have length > 3
        Parameters:
        ---------------
        col: str
            Movie title, ex. Toy Story (1995);3
        df_movies: pandas df 
            Original utility matrix, 
        df_moviesplots: pandas df 
            Movie-Plot DF, 
        df_utilitymatrix: pandas df 
            New utility matrix
    '''
    def get_plot_from_omdb(self, col, df_movies, df_moviesplots, df_utilitymatrix):
        original_title = col.split(';')[0] #Movie title

        title = original_title[:-6].strip();
        year = original_title[-5:-1];
        #Add a dot at the end of the movie
        #print("Title", title, "Year", year)
        plot = ' '.join(title.split(' ')) + '. '

        url = "http://www.omdbapi.com/?t=" + title + "&y" + year + "&plot=full&r=json" + "&apikey=bb9d0462"

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2049.0 Safari/537.36"}
        r = requests.get(url, headers = headers);
        jsondata = json.loads(r.content.decode());
        if('Plot' in jsondata): #if the movie was found in the omdb;

            #Store plot + title
            plot += jsondata['Plot']
            #print(plot)

        if plot != None and plot != '' and plot!= np.nan and len(plot)>3:
            df_moviesplots.loc[len(df_moviesplots)] = [original_title, plot]; #original_title, clean_title + plot
            # df_utilitymatrix == df_movies if len(plot) for all plots > 3
            df_utilitymatrix[col] = df_movies[col];
            print(len(df_utilitymatrix.columns));

        return df_moviesplots, df_utilitymatrix;

    def handle(self, *args, **options):
        path_utility_matrix = options['umatrixfile'];
        df_movies = pd.read_csv(path_utility_matrix);
        movieslist = list(df_movies.columns[1:]);

        df_movies_plots = pd.DataFrame(columns=['title', 'plot']);
        df_utility_matrix = pd.DataFrame();
        #df_movies['user']: List of users from 1 to n_users
        df_utility_matrix['user'] = df_movies['user'];

        print('nmovies', len(movieslist))
        for m in movieslist[:]:
            self.get_plot_from_omdb(m, df_movies, df_movies_plots, df_utility_matrix);

        print(len(df_movies.columns), '---', len(df_utility_matrix.columns))
        outputfile = options['plotsfile'];
        df_movies_plots.to_csv(outputfile, index=False);
        output_matrix_file = options['umatrixoutfile'];
        df_utility_matrix.to_csv(output_matrix_file, index=False);
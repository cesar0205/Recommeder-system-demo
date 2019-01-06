from django.db import models
from django.contrib.auth.models import User
import jsonfield
import json
import numpy as np

# Create your models here.
class UserProfile(models.Model):
    #If the user User is deleted then also detele its profile
    user = models.ForeignKey(User, unique = True, on_delete= models.CASCADE)
    #Stores all movie ratings from the user
    array = jsonfield.JSONField(default=list);   #Movie ratings list from user (in string format)
    arrayratedmoviesindxs = jsonfield.JSONField(default=list); #Movie indexes list for movies rated by user (in string format)
    name = models.CharField(max_length=1000)
    lastrecs = jsonfield.JSONField(default=list); #The last recommedations for the user

    def __unicode__(self):
        return self.name;

    #Saves the last recomendations associated with the user
    def save(self, *args, **kwargs):
        create = kwargs.pop('create', None);
        recsvec = kwargs.pop('recsvec', None);
        print("create: ", create)
        if(create == True):
            super(UserProfile, self).save(*args, **kwargs);
        elif(recsvec is not None):
            self.lastrecs = json.dumps(recsvec.tolist())
            super(UserProfile, self).save(*args, **kwargs);
        else:
            nmovies = MovieData.objects.count();

            array = np.zeros(nmovies);
            ratedmovies = self.ratedmovies.all();  #Returns all MovieRated objects rated by the user
            #String representation with the movie index of movies rated by user
            self.arrayratedmoviesindxs = json.dumps([m.movieindx for m in ratedmovies]);

            for m in ratedmovies:
                array[m.movieindx] = m.value;
            self.array = json.dumps(array.tolist())
            super(UserProfile, self).save(*args, **kwargs);

#A MovieRated is associated with one and only one UserProfile. A MovieRated instance is created each time a user
#wants to rate a movie. So MovieRated is not the same as MovieData, as MovieData can live without a user.
class MovieRated(models.Model):
    #ratedmovies is the collection that UserProfile uses to get a reference of all MovieRated objects related to it.
    user = models.ForeignKey(UserProfile, related_name='ratedmovies', on_delete= models.CASCADE);
    movie = models.CharField(max_length = 100);
    movieindx = models.IntegerField(default = -1);
    value = models.IntegerField();

class MovieData(models.Model):
    title = models.CharField(max_length = 100);
    array = jsonfield.JSONField();
    ndim = models.IntegerField(default = 300);
    description = models.TextField();

from django.contrib import admin

# Register your models here.
from recommender_system_app.models import UserProfile, MovieRated, MovieData

admin.site.register(UserProfile)
admin.site.register(MovieRated)
admin.site.register(MovieData)

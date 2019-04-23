# Recommender-system-web-application

Web application that features two machine learning applications.

###Recommender system for movies.

Users register into the application and rate movies that internally helps 
a Colaborative Filtering Item-Based model (and optionally a LogLikelihood 
model) to learn users’ preferences. 
The system updates automatically the model and gives recommendations to 
other users.

###Bayesian A/B testing for banners.

A module was developed to test the popularity of a series of banners.
A bayesian A/B testing model is updated after each user click on a banner
and their display frequencies change depending on the state of the model.

####-Framework details.

This engine is written in Python, Scikit-Learn on top of NumPy and SciPy stack. It uses Django for webserver backend.

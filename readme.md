# CS50’s Introduction to Artificial Intelligence with Python
# Project 4: Learning: Shopping Prediction

Aim: Write an AI to predict whether online shopping customers will complete a purchase.

Description: When users are shopping online, not all will end up purchasing something. 
It might be useful, though, for a shopping website to be able to predict whether a user 
intends to make a purchase or not, in order to show perhaps a different behaviour. That's 
where machine leaning will come in.

The task in this problem is to build a nearset-neighbor classifier. Given information 
about a user -how many pages they've visited, the machine they are using, etc. - the 
classifier should predict if the user will make a purchase. The classifier won't ever 
be perfectly accurate (it's not easy modeling human behaviour!) but it should better 
than guessing at random.

The data for this project is from a shopping website with about 12,000 users sessions.
The package *scikt-learn* is used, and in particular the *KNeighborsClassifier* model. 
A true positive rate (sensitivity) of 40% and true negative rate (specifity) of 90% is 
achieved. Our goal is to build a classifier that performs reasonably on both metrics.

See full description here: https://cs50.harvard.edu/ai/2020/projects/4/shopping/

UPDATE 1: A *DecisionTreeClassifier* model was used, proving more useful for this 
problem by getting better predictions while taking much shorter time to train. A sensitivity of 59% and 
a specifity of 94% is achieved after fine tuning the model using Grid Search.

When ran in interactive window, the command line arguments can be set with sys.argv=[...], 
and theresults (models, variables) can be retrieved from the dict main_locals, e.g. 
*main_locals['gs'].best_estimator_* to get the optimized model or *main_locals['sorted_scores'][:5]* 
to see grid search scores.

UPDATE 2: A *RandomForestClassifier* model was used to improve the results slightly further.
In this case, grid search does not seem to be able to improve the results anymore, so
default parameters are used to obtain a sensitivity of 57% and a specifity of 96%. Since
the test data is split at random, these results vary ever so slighly, but consistent
accuracy of around 90% (right predictions in total) is obtained for the test data.

This model also provides a feature importance estimation, which is very useful in
assesing which features are most relevant to predict the outcome. In this case, the
most important features to predict a purchase are estimated as follows:

Model estimated feature importances:
PageValues              0.3982
ProductRelated_Duration 0.0876
ExitRates               0.0839
ProductRelated          0.0712
Administrative_Duration 0.0561
BounceRates             0.0524
Month                   0.0442
Administrative          0.0426
Region                  0.0303
TrafficType             0.0296
Informational_Duration  0.026
Browser                 0.0191
OperatingSystems        0.0172
Informational           0.0168
VisitorType             0.0113
Weekend                 0.01
SpecialDay              0.0035

PageValues, ExitRates and BounceRates are values calculated by Google Analytics. PageValues
is clearly and by far the most important feature in this case. The page values of Google 
Analytics is intended to give us an idea of which pages in our site contribute more to 
the site's revenue, so it is reasonable that it is in first place. Other important features 
seem to be product related duration and pages, if the user spends also time on administrative
pages (user account,...), the month or even the region. Special days, os and browser or visitor
type seem to be the least important in our predictions.


Usage: Usage: python shopping.py data [model] [--options]
              model: {knn, tree}, default='knn' where:
              knn: k-nearest neighbors classifier
              tree: decision tree classifier
              forest: random forest classifier
              Options:
              --grid-search: activates grid search (will take longer)

Example:
```
$ python shopping.py shopping.csv tree
Results for chosen model (DecisionTreeClassifier()) with default parameters:
Correct: 4252
Incorrect: 680
True Positive Rate: 60.24%
True Negative Rate: 90.89%
```

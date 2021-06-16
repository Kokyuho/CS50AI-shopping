# CS50’s Introduction to Artificial Intelligence with Python
# Project 4: Learning: Shopping Prediction

Aim: Write an AI to predict whether online shopping customers will complete a purchase.

Description: When users are shopping online, not all will end up purchasing something. It might be useful, though, for a shopping website to be able to predict whether a user intends to make a purchase or not, in order to show perhaps a different behaviour. That's where machine leaning will come in.

The task in this problem is to build a nearset-neighbor classifier. Given information about a user -how many pages they've visited, the machine they are using, etc. - the classifier should predict if the user will make a purchase. The classifier won't ever be perfectly accurate (it's not easy modeling human behaviour!) but it should better than guessing at random.

The data for this project is from a shopping website with about 12,000 users sessions.The package *scikt-learn* is used, and in particular the *KNeighborsClassifier* model. A true positive rate (sensitivity) of 40.4% and true negative rate (specifity) of 90.3% is achieved. Our goal is to build a classifier that performs reasonably on both metrics.

See full description here: https://cs50.harvard.edu/ai/2020/projects/4/shopping/

Usage: python shopping.py shopping.csv

Example:
```
$ python shopping.py shopping.csv
Correct: 4088
Incorrect: 844
True Positive Rate: 41.02%
True Negative Rate: 90.55%
```

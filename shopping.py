import csv
from re import X
import sys
import calendar
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

TEST_SIZE = 0.4

def main():

    # Check command-line arguments for data (sys.argv[1])
    if len(sys.argv) < 2:
        sys.exit("Usage: python shopping.py data [model] [--options]\n"
                  "model: {knn, tree}, default='knn' where:\n"
                  "knn: k-nearest neighbors classifier\n"
                  "tree: decision tree classifier\n"
                  "forest: random forest classifier\n"
                  "Options:\n"
                  "--grid-search: activates grid search (will take longer)")

    # Check if model was given (sys.argv[2]), else use knn (K-nearest neighbors)
    model = 'knn'
    if len(sys.argv) > 2:
        model = sys.argv[2]
        if model in ('knn', 'tree', 'forest'):
            pass
        else:
            sys.exit("Usage: python shopping.py data [model] [--options]\n"
                     "model: {knn, tree}, default='knn' where:\n"
                     "knn: k-nearest neighbors classifier\n"
                     "tree: decision tree classifier\n"
                     "forest: random forest classifier\n"
                     "Options:\n"
                     "--grid-search: activates grid search (will take longer)")

    # Check command line arguments to see if option --grid-search was given
    if len(sys.argv) > 2:
        for argv in sys.argv:
            if argv == '--grid-search':
                _grid_search = True
            else:
                _grid_search = False

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Create transformation pipeline
    pipe = Pipeline([
        ('std_scaler', StandardScaler()),
    ])

    # Train chosen model with dafault parameters
    model = train_model(X_train, y_train, model, pipe)

    # Make predictions on test data and evaluate results
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Results for chosen model ({model[-1].__class__()}) with default parameters:")
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%\n")

    # Run grid search to optimize model if selected
    if _grid_search:
        print("Now running grid search optimization...")
        gs = grid_search(model, X_train, y_train)
        model_gs = gs.best_estimator_
    
        # Make predictions on test data and evaluate results
        predictions = model_gs.predict(X_test)
        sensitivity, specificity = evaluate(y_test, predictions)

        # Print results
        print(f"Results for chosen model ({model[-1].__class__()}) after grid search optimization:")
        print(f"Correct: {(y_test == predictions).sum()}")
        print(f"Incorrect: {(y_test != predictions).sum()}")
        print(f"True Positive Rate: {100 * sensitivity:.2f}%")
        print(f"True Negative Rate: {100 * specificity:.2f}%\n")

        # Sort and store grid search scores
        scores = []
        cvres = gs.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            scores.append((np.sqrt(-mean_score), params))
        sorted_scores = sorted(scores, key=lambda score: score[0])

    # If model is forest, process and store feature_importances_ estimation
    if type(model[-1]) is RandomForestClassifier:

        # Read headers
        with open(sys.argv[1]) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                headers = row
                break
        
        # Get feature importances values array
        importances = model[-1].feature_importances_

        # Zip and sort
        feature_importances = list(zip(headers, importances))
        feature_importances = sorted(feature_importances, 
                                     key= lambda feature: feature[1],
                                     reverse=True)

        # Print feature importances
        print("Model estimated feature importances:")
        for name, value in feature_importances:
            print("{:<24}{}".format(name, round(value,4)))

    return locals()


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)

        evidence = []
        labels = []

        # Skip header
        next(reader, None)

        # Fill evidence and labels data
        for row in reader:

            # Create base evidence row
            evidence_row = [
                int(row[0]), float(row[1]), int(row[2]), float(row[3]), int(row[4]), 
                float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), 
                row[10], int(row[11]), int(row[12]), int(row[13]), int(row[14]), 
                row[15], row[16]]
            
            # Convert calendar month abbreviation to integer
            abbr_to_num = {name: num for num, name in enumerate(calendar.month_abbr) if num}
            abbr_to_num['June'] = abbr_to_num['Jun']
            evidence_row[10] = int(abbr_to_num[row[10]] - 1)

            # Return visitor integer conversion
            evidence_row[15] = 1 if row[15] == 'Returning_Visitor' else 0

            # Weekend integer conversion
            evidence_row[16] = 0 if row[16] == 'FALSE' else 1

            # Append row to evidence list
            evidence.append(evidence_row)

            # Create and append label
            labels_row = 0 if row[17] == 'FALSE' else 1
            labels.append(labels_row)
        return (evidence, labels)


def train_model(evidence, labels, model, pipe):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    if model == 'knn':
        pipe.steps.append(
            ('knn', KNeighborsClassifier(n_neighbors=1, 
                                         weights='uniform', 
                                         algorithm='auto', 
                                         leaf_size=30, 
                                         p=2, 
                                         metric='minkowski', 
                                         metric_params=None, 
                                         n_jobs=None, 
                                         ))
        )

    elif model == 'tree':
        pipe.steps.append(
            ('tree', DecisionTreeClassifier(criterion='gini', 
                                           splitter='best', 
                                           max_depth=None, 
                                           min_samples_split=2, 
                                           min_samples_leaf=1, 
                                           min_weight_fraction_leaf=0.0, 
                                           max_features=None, 
                                           random_state=None, 
                                           max_leaf_nodes=None, 
                                           min_impurity_decrease=0.0, 
                                           class_weight=None, 
                                           ccp_alpha=0.0))
        )

    elif model == 'forest':
        pipe.steps.append(
            ('forest', RandomForestClassifier(n_estimators=100,
                                              criterion='gini',
                                              max_depth=None,
                                              min_samples_split=2,
                                              min_samples_leaf=1,
                                              min_weight_fraction_leaf=0.0,
                                              max_features='auto',
                                              max_leaf_nodes=None,
                                              min_impurity_decrease=0.0,
                                              min_impurity_split=None,
                                              bootstrap=True,
                                              oob_score=False,
                                              n_jobs=-1,
                                              random_state=None,
                                              verbose=0,
                                              warm_start=False,
                                              class_weight=None,
                                              ccp_alpha=0.0,
                                              max_samples=None))
        )
     
    model = pipe.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    sensitivity = 0
    specifity = 0
    n = 0
    n_pos = 0
    n_neg = 0

    for label in labels:
        if label == 1:
            n_pos += 1
            if predictions[n] == 1:
                sensitivity += 1
        elif label == 0:
            n_neg += 1
            if predictions[n] == 0:
                specifity += 1
        n += 1
    
    sensitivity = sensitivity/n_pos
    specifity = specifity/n_neg

    return (sensitivity, specifity)

def grid_search(pipe, X_train, y_train):
    """
    Takes as input the estimator (i.e. the model pipeline) and X_train data and 
    y_train labels. Returns the optimized model.
    """
    if type(pipe[-1]) is KNeighborsClassifier:
        param_grid = [
            dict(std_scaler=['passthrough', StandardScaler()],
                 knn__n_neighbors=[1, 5, 10],
                 knn__weights=['uniform', 'distance'],
                 knn__algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'],
                 knn__p=[1, 2]),
        ]

    elif type(pipe[-1]) is DecisionTreeClassifier:
        param_grid = [
            dict(std_scaler=['passthrough', StandardScaler()],
                 tree__criterion=['gini', 'entropy'], 
                 tree__max_depth=[3, 5, 10],
                 tree__min_samples_split=[2, 5, 10],
                 tree__min_samples_leaf=[5, 10, 15]),
        ]

    elif type(pipe[-1]) is RandomForestClassifier:
        param_grid = [
            dict(std_scaler=['passthrough', StandardScaler()],
                 forest__n_estimators=[50, 100, 200],
                 forest__criterion=['gini', 'entropy'], 
                 forest__max_depth=[5, 10, None],
                 forest__min_samples_leaf=[1, 5, 10],
                 forest__max_features=['auto', 'sqrt'],
                 forest__bootstrap=[True, False],
                 ),
        ]
    
    grid_search = GridSearchCV(pipe, 
                               param_grid, 
                               scoring='neg_mean_squared_error', #default=None
                               n_jobs=-1, # can be set to -1 for parallel processing
                               refit=True, 
                               cv=None, # Default 5-fold cross-validation, 
                               verbose=1,
                               pre_dispatch='2*n_jobs', 
                               return_train_score=False)

    grid_search.fit(X_train, y_train)

    return grid_search

if __name__ == "__main__":
    main_locals = main()
    # model, X_train, X_test, y_train, y_test = main()

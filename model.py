"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
def impurity(labels):
    """Return a non-negative impurity score for a 1D array of integer class labels."""
    if len(labels) == 0:
        return 0.0 

    _, counts = np.unique(labels , return_counts = True )
    probabilities = counts / len(labels)


    return float(1 - np.sum(probabilities**2))

# Step 2 - split_dataset
import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    # TODO: partition rows into left (feature <= threshold) and right (feature > threshold)
    mask = features[:, feature_index] <= threshold

    left_features = features[mask]
    left_labels = labels[mask]

    right_features = features[~mask]
    right_labels = labels[~mask]

    return left_features, left_labels, right_features, right_labels

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    # TODO: return a score where higher means the children are purer than the parent.
    n = len(parent_labels)
    l_w = len(left_labels)/n 
    r_w = len(right_labels)/n
    
    sr = impurity(parent_labels) -(l_w*impurity(left_labels) + r_w*impurity(right_labels))
    return sr

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices):
    # TODO: search feature_indices for the (feature, threshold) that best improves purity.
    best_feature =  None
    best_threshold = None 
    best_score = 0.0 

    for feature_index  in feature_indices :
        values = np.unique(features[:,feature_index])
        
        thresholds = (values[:-1]+ values[1:])/2
        
        for threshold in thresholds :

            left_features , left_labels , right_features , right_labels = split_dataset(features , labels , feature_index , threshold)

            if len(left_labels )==0 or len(right_labels) ==0 :
                continue 

            score =split_score(
                labels, left_labels, right_labels
            )

            if score > best_score:
                best_score = score
                best_feature = feature_index
                best_threshold = threshold

    return {
        'feature_index': best_feature,
        'threshold': best_threshold,
        'score': float(best_score)
    }

# Step 5 - should_stop
def should_stop(labels, depth, max_depth, min_samples_split):
    """Return True if this node should become a leaf instead of splitting further."""
    # TODO: decide whether to stop growing based on purity, depth, and size...
    if len(np.unique(labels))==1 :
        return True 

    if depth >= max_depth:
        return True 

    if len(labels) < min_samples_split:
        return True 

    return False

# Step 6 - leaf_prediction
def leaf_prediction(labels):
    # TODO: choose a single class label to output for a leaf given the labels that reached it
    values , counts = np.unique(labels , return_counts = True )

    majority_class = values[np.argmax(counts)]

    return int(majority_class)

# Step 7 - build_tree
import numpy as np

def build_tree(features, labels, max_depth=10,
               min_samples_split=2, feature_subset=None, depth=0):

    # Step 1: Check stopping conditions
    if should_stop(labels, depth, max_depth, min_samples_split):
        return {
            'leaf': True,
            'prediction': leaf_prediction(labels)
        }

    # Step 2: Select allowed features
    if feature_subset is None:
        feature_indices = range(features.shape[1])
    else:
        feature_indices = feature_subset

    # Step 3: Find best split
    best = best_split(features, labels, feature_indices)

    # Step 4: Check if a valid split exists
    if best['feature_index'] is None or best['score'] <= 0:
        return {
            'leaf': True,
            'prediction': leaf_prediction(labels)
        }

    feature_index = best['feature_index']
    threshold = best['threshold']

    # Step 5: Split dataset
    left_features, left_labels, right_features, right_labels = \
        split_dataset(features, labels, feature_index, threshold)

    # Step 6: Recursively build child nodes
    left_node = build_tree(
        left_features,
        left_labels,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        feature_subset=feature_subset,
        depth=depth + 1
    )

    right_node = build_tree(
        right_features,
        right_labels,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        feature_subset=feature_subset,
        depth=depth + 1
    )

    # Step 7: Return internal node
    return {
        'leaf': False,
        'feature_index': int(feature_index),
        'threshold': float(threshold),
        'left': left_node,
        'right': right_node
    }

# Step 8 - predict_example_tree
def predict_example_tree(tree, example):
    # TODO: walk the example down the fitted tree until you reach a leaf, then return its prediction.
    while not tree['leaf']:

        feature_index = tree['feature_index']
        threshold = tree['threshold']

        # step 3 decide left or right branch 

        if example[feature_index]<= threshold:
            tree= tree['left']
        else :
            tree = tree['right']

    return int(tree['prediction'])

# Step 9 - predict_tree
def predict_tree(tree, features):
    """Predict class labels for every row of `features` using a fitted decision tree.

    tree: dict returned by build_tree
    features: np.ndarray of shape (n, d)
    returns: np.ndarray of shape (n,) with integer class labels
    """
    # TODO: return predicted class for each row of features using the fitted tree.
    predictions = []

    for example in features :
        prediction = predict_example_tree(tree , example)
        predictions.append(prediction)

    return np.array(predictions , dtype = int )

# Step 10 - bootstrap_sample (not yet solved)
# TODO: implement

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement


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

# Step 5 - should_stop (not yet solved)
# TODO: implement

# Step 6 - leaf_prediction (not yet solved)
# TODO: implement

# Step 7 - build_tree (not yet solved)
# TODO: implement

# Step 8 - predict_example_tree (not yet solved)
# TODO: implement

# Step 9 - predict_tree (not yet solved)
# TODO: implement

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


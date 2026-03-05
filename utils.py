import warnings
import copy
import sys
import os
import contextlib
import lightgbm as lgb
import pickle, hashlib
import numpy as np
from scipy.sparse import csr_matrix
from collections import OrderedDict
import sklearn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
from packaging import version




def efficient_append_column(S, A):
    A = A.reshape(len(S), -1)
    return np.hstack([S,A])
    
    if sp.issparse(S):
        # Convert the dense array A into a sparse matrix with a single column
        S = S.toarray()
    n = S.shape[0]
    dim = S.shape[1]
    dim_extra = A.reshape(n, -1).shape[1]

    # Pre-allocate a large array that will hold S_embedding, A, and the ones_like(A)
    SA = np.empty((n, dim + dim_extra))

    # Fill in the S_embedding part
    SA[:, :-1] = S
    SA[:, -1] = A
    return SA


def get_sparse_param():
    # Check the sklearn version
    sklearn_version = sklearn.__version__ 
    print(sklearn_version)
    if version.parse(sklearn_version) <= version.parse("1.0.2"):
        return {"sparse": True}
    else:
        # For versions >= 0.24, use 'sparse'
        return {"sparse_output": True}



def split_with_row_id(n, test_size=0.2, seed=None, row_id=None, train_indices=None):
    """
    Splits indices into training and validation sets. If `row_id` is provided, 
    it ensures that rows with the same `row_id` stay together in the same set.
    
    Parameters:
    - n: Total number of rows.
    - test_size: Proportion of the dataset to include in the validation split.
    - seed: Random seed for reproducibility.
    - row_id: List or array of `row_id`s for each row. If `None`, treat each row independently.
    - train_indices: Predefined training indices (if available). If `None`, a new split is made.
    
    Returns:
    - train_indices: Array of indices for the training set.
    - val_indices: Array of indices for the validation set.
    """
    indices = np.arange(n)
    
    if row_id is None:
        # If no row_id is provided, do a regular train/val split
        if train_indices is None:
            train_indices, val_indices = train_test_split(indices, test_size=test_size, random_state=seed)
        else:
            val_indices = np.setdiff1d(indices, train_indices)
    else:
        # If row_id is provided, ensure that all rows with the same row_id stay together
        df = pd.DataFrame({'index': indices, 'row_id': row_id})
        
        # Group by row_id and take a unique index for each row_id
        unique_row_id_groups = df.groupby('row_id')['index'].apply(list).reset_index()
        
        # Sample row_id groups for validation
        if train_indices is None:
            # Perform train/test split at the row_id level
            unique_row_ids = unique_row_id_groups['row_id'].values
            train_row_ids, val_row_ids = train_test_split(unique_row_ids, test_size=test_size, random_state=seed)
            
            # Get the corresponding indices for each set of row_ids
            train_indices = np.concatenate(unique_row_id_groups[unique_row_id_groups['row_id'].isin(train_row_ids)]['index'].values)
            val_indices = np.concatenate(unique_row_id_groups[unique_row_id_groups['row_id'].isin(val_row_ids)]['index'].values)
        else:
            # If train_indices is provided, infer val_indices based on row_ids
            train_row_ids = df[df['index'].isin(train_indices)]['row_id'].unique()
            val_indices = df[~df['row_id'].isin(train_row_ids)]['index'].values
    
    return train_indices, val_indices



 

class LRUCache:
    """
    A class that implements a Least Recently Used (LRU) cache using an OrderedDict.
    
    Attributes:
    -----------
    cache : OrderedDict
        The cache that stores key-value pairs with keys ordered by usage.
    max_size : int
        The maximum number of items the cache can hold.
    
    Methods:
    --------
    get(key):
        Retrieves the value associated with the given key if it exists in the cache.
        Moves the key to the end to indicate recent usage.
        Returns None if the key is not present.
        
    put(key, value):
        Inserts a new key-value pair into the cache. If the key already exists,
        updates its value and moves it to the end to indicate recent usage.
        If the cache exceeds the maximum size, it removes the least recently used item.
    
    __contains__(key):
        Checks if a given key exists in the cache.
    """
    
    def __init__(self, max_size=100):
        """
        Initializes the LRUCache with a given maximum size.
        
        Parameters:
        -----------
        max_size : int, optional
            The maximum number of items the cache can hold (default is 100).
        """
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key):
        """
        Retrieves the value associated with the given key from the cache.
        
        Parameters:
        -----------
        key : Any
            The key to retrieve the associated value for.
        
        Returns:
        --------
        Any
            The value associated with the key if it exists, otherwise None.
        """
        if key in self.cache:
            # Move the accessed key to the end to show that it was recently used
            self.cache.move_to_end(key)
            return self.cache[key]
        else:
            return None

    def put(self, key, value):
        """
        Inserts or updates a key-value pair in the cache.
        
        Parameters:
        -----------
        key : Any
            The key to be inserted or updated.
        value : Any
            The value to be associated with the key.
        """
        if key in self.cache:
            # Update the value and move it to the end
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            # Pop the first item (the least recently used one)
            self.cache.popitem(last=False)

    def __contains__(self, key):
        """
        Checks if a key exists in the cache.
        
        Parameters:
        -----------
        key : Any
            The key to check for existence in the cache.
        
        Returns:
        --------
        bool
            True if the key exists in the cache, False otherwise.
        """
        return key in self.cache


class StratifiedModel:
    """
    A class that represents a stratified model consisting of two sub-models
    for different strata of the data. This model is used to predict outcomes
    based on a feature matrix that includes both covariates and a binary 
    indicator for the stratum.
    
    Attributes:
    -----------
    model_0 : Any
        The model used to predict outcomes when the binary indicator (A) is 0.
    model_1 : Any
        The model used to predict outcomes when the binary indicator (A) is 1.
    
    Methods:
    --------
    predict(X):
        Predicts the outcome based on the covariates and the binary indicator
        by combining predictions from both sub-models.
    """
    
    def __init__(self, model_0, model_1, trained_with_offset = False):
        """
        Initializes the StratifiedModel with two sub-models.
        
        Parameters:
        -----------
        model_0 : Any
            The model for predicting outcomes when the binary indicator is 0.
        model_1 : Any
            The model for predicting outcomes when the binary indicator is 1.
        """
        self.model_0 = model_0
        self.model_1 = model_1
        self.trained_with_offset = trained_with_offset

    def predict(self, X, A=None):
        """
        Predicts the outcome based on the covariates and a binary indicator.
    
        Parameters:
        -----------
        X : np.ndarray
            The feature matrix where the last column represents the binary indicator (A),
            and the remaining columns represent the covariates (S).
    
        Returns:
        --------
        np.ndarray
            The predicted outcomes based on the binary indicator and the sub-models.
        """
        if A is None:
            S = X[:, :-1]  # Covariates (all columns except the last one)
            A = X[:, -1]   # Binary indicator (last column)
        else:
            S = X
        if np.all(A == 1):
            # If all A is 1, use only model_1
            mu_1 = self.model_1.predict(S)
            if self.trained_with_offset:
                mu_1 = mu_1 + self.model_0.predict(S)
            return mu_1
        elif np.all(A == 0):
            # If all A is 0, use only model_0
            return self.model_0.predict(S)
        else:
            # If A contains both 0 and 1, combine the predictions
            mu_1 = self.model_1.predict(S)  # Predictions when A == 1
            mu_0 = self.model_0.predict(S)  # Predictions when A == 0
            if self.trained_with_offset:
                mu_1 = mu_0 + mu_1
            mu = A * mu_1 + (1 - A) * mu_0  # Combine the predictions based on A
            return mu




# hack to mute lightgbm messages from each training iteration
@contextlib.contextmanager
def suppress_lightgbm_output(verbose=False):
    """
    Context manager to suppress LightGBM output.
    
    If verbose is False, it redirects stdout and stderr to os.devnull
    to suppress LightGBM messages and warnings. If verbose is True,
    it does nothing.
    """
    if verbose:
        yield
    else:
        # Redirect stdout and stderr to devnull to suppress LightGBM messages and warnings
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        try:
            yield
        finally:
            # Restore original stdout and stderr
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = original_stdout
            sys.stderr = original_stderr

# Convenience function to subset training and validation folds of many arrays
def subset_by_indices(*arrays, indices):
    """
    Subset multiple arrays by indices.
    
    Parameters:
    *arrays : list of arrays
        The arrays to be subsetted.
    indices : list or array
        The indices to subset by.
    
    Returns:
    list of arrays
        The subsetted arrays.
    """
    subsets = []
    for array in arrays:
        if isinstance(array, csr_matrix):
            subsets.append(array[indices, :])
        else:
            subsets.append(array[indices])
    if len(subsets) == 1:
        subsets = subsets[0]
    return subsets


 

def ensure_list_values(param_grid):
    param_grid = copy.deepcopy(param_grid)
    keys_to_remove = [key for key in param_grid]
    for key in keys_to_remove:
        if param_grid[key] is None:
            param_grid.pop(key)
        elif key == "interaction_constraints":
            if not isinstance(param_grid[key][0][0], (list, tuple)):
                param_grid[key] = [param_grid[key]]
        else:
            if not isinstance(param_grid[key], (list, tuple)):
                param_grid[key] = [param_grid[key]]
    return param_grid



def split_preds(preds, num_groups):
    """
    Split predictions into multiple groups.
    
    Parameters:
    preds : array
        The predictions to be split.
    num_groups : int
        The number of groups to split into.
    
    Returns:
    tuple of arrays
        The split predictions.
    """
    n = len(preds) // num_groups
    return tuple(preds[i*n:(i+1)*n] for i in range(num_groups))

def check_weights(weights, n):
        if weights is None:
            return np.ones(n) 
        else:
            weights = weights * (n/np.sum(weights))
            return np.array(weights).reshape(-1)

def generate_hash_numpy_array(array, hash_function='sha256'):
        """
        Generate a unique hash code for a NumPy array.

        Parameters:
        array (numpy.ndarray): The NumPy array to hash.
        hash_function (str): The hash function to use ('md5', 'sha1', 'sha256', etc.).

        Returns:
        str: The hexadecimal hash code of the array.
        """
        
        num_rows = min(100, array.shape[0])
        array_bytes_1 = array[:num_rows].tobytes()
        array_bytes_0 = array[num_rows:].tobytes()
   

        # Create a new hash object using the specified hash function
        hash_obj = hashlib.new(hash_function)

        # Update the hash object with the byte stream and shape bytes
        hash_obj.update(array_bytes_1)
        hash_obj.update(array_bytes_0)
        hash_obj.update(np.asarray(array.shape).tobytes())

        # Return the hexadecimal representation of the hash
        return hash_obj.hexdigest()


def generate_hash(obj, hash_function='sha256'):
    """
    Generate a unique hash code for an arbitrary object.

    Parameters:
    obj (any): The object to hash.
    hash_function (str): The hash function to use ('md5', 'sha1', 'sha256', etc.).

    Returns:
    str: The hexadecimal hash code of the object.
    """
    # Serialize the object to a byte stream
    obj_bytes = pickle.dumps(obj)

    # Create a new hash object using the specified hash function
    hash_obj = hashlib.new(hash_function)

    # Update the hash object with the byte stream
    hash_obj.update(obj_bytes)

    # Return the hexadecimal representation of the hash
    return hash_obj.hexdigest()

def shrink_leaf_values(model: lgb.Booster, shrinkage_factor: float, init_model = None) -> lgb.Booster:
    """
    Shrinks the leaf values of a LightGBM model by a specified factor using the set_leaf_output method.

    Parameters:
    model (lgb.Booster): The original trained LightGBM model.
    init_model (lgb.Booster): The initial LightGBM model used for training the new model.
    shrinkage_factor (float): The factor by which to shrink the leaf values. It should be between 0 and 1.

    Returns:
    lgb.Booster: A new LightGBM model with shrunk leaf values.
    """
    # Create a deep copy of the original model to ensure the original model is not modified
    new_model = copy.deepcopy(model)
    
    # Get the number of trees in the model
    num_trees = new_model.num_trees()
    if init_model is None:
        num_trees_previous = 0
    else:
        num_trees_previous = init_model.num_trees()
    
    # Iterate over all new trees added after the init_model
    for tree_id in range(num_trees_previous, num_trees):
        # Get the tree structure
        tree_structure = new_model.dump_model()['tree_info'][tree_id]['tree_structure']
        
        # Define a recursive function to traverse the tree and shrink leaf values
        def shrink_leaves(node):
            if 'leaf_value' in node:
                # Shrink the leaf value
                current_value = node['leaf_value']
                new_value = current_value * shrinkage_factor
                new_model.set_leaf_output(tree_id, node['leaf_index'], new_value)
            else:
                # Traverse left and right children
                if 'left_child' in node:
                    shrink_leaves(node['left_child'])
                if 'right_child' in node:
                    shrink_leaves(node['right_child'])
        
        # Start the recursion from the root node
        shrink_leaves(tree_structure)
    
    return new_model
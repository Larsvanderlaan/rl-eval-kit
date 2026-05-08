import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from utils import *
import utils as utils
import json
import copy
from tqdm import tqdm
import time
import random

 
def fit_fqe_boosted(
    S,
    A,
    S_next,
    A_next,
    rewards,
    discount_factor,
    is_terminal_outcome=None,
    weights=None,
    categorical_feature=None,
    lgb_params=None,
    init_model=None,
    init_score=None,
    test_size=0.2,
    train_indices=None,
    data_dict_test=None,
    refit_on_all_data=True,
    boost=True,
    fit_control=None,
    seed=42,
    verbose=False,
):
    """
    Wrapper that implements fitted Q-iteration (FQI/FQE) by calling
    `fit_value_iteration_boosted` on augmented state-action features.

    We model:
        Q(s, a) = V( x ), where x = concat( state_features(s), action_features(a) )

    and pass:
        X      = concat(S, A)       as `S` to fit_value_iteration_boosted
        X_next = concat(S_next, A_next) as `S_next`

    Parameters
    ----------
    S : array-like, shape (n, d_s) or (n,)
        Current states S_t.

    A : array-like, shape (n, d_a) or (n,)
        Current actions A_t.

    S_next : array-like, shape (n, d_s) or (n,)
        Next states S_{t+1}.

    A_next : array-like, shape (n, d_a) or (n,)
        Next actions A'_{t+1} (e.g., drawn from the evaluation policy for FQE).

    rewards : array-like, shape (n,)
        One-step rewards R_t.

    discount_factor : float
        Discount factor γ ∈ [0, 1).

    is_terminal_outcome, weights, categorical_feature, lgb_params, init_model,
    init_score, test_size, train_indices, data_dict_test, refit_on_all_data,
    boost, fit_control, seed, verbose :
        Passed through to `fit_value_iteration_boosted`.

    Returns
    -------
    dict
        Whatever `fit_value_iteration_boosted` returns, but now interpreting
        the model as a Q-function in the concatenated (S, A) space.
    """
    # Ensure arrays
    S = np.asarray(S)
    A = np.asarray(A)
    S_next = np.asarray(S_next)
    A_next = np.asarray(A_next)
    rewards = np.asarray(rewards).reshape(-1)

    n = rewards.shape[0]

    # Flatten state and action features and concatenate along feature dimension
    S_flat = S.reshape(n, -1)
    A_flat = A.reshape(n, -1)
    X = np.concatenate([S_flat, A_flat], axis=1)          # (n, d_x)

    S_next_flat = S_next.reshape(n, -1)
    A_next_flat = A_next.reshape(n, -1)
    X_next = np.concatenate([S_next_flat, A_next_flat], axis=1)

    # Forward to your boosted FVI, now acting as FQI/FQE on (S,A)
    result = fit_fvi_boosted(
        S=X,
        S_next=X_next,
        rewards=rewards,
        discount_factor=discount_factor,
        is_terminal_outcome=is_terminal_outcome,
        weights=weights,
        categorical_feature=categorical_feature,
        lgb_params=lgb_params,
        init_model=init_model,
        init_score=init_score,
        test_size=test_size,
        train_indices=train_indices,
        data_dict_test=data_dict_test,
        refit_on_all_data=refit_on_all_data,
        boost=boost,
        fit_control=fit_control,
        seed=seed,
        verbose=verbose,
    )

    return result




def fit_fvi_boosted(S, S_next, rewards, discount_factor, is_terminal_outcome=None, weights=None,
                        categorical_feature=None, lgb_params=None, init_model=None, init_score=None,
                        test_size=0.2, train_indices=None, data_dict_test=None, refit_on_all_data=False,
                        boost=True, fit_control=None, seed=42, verbose=False):
    """
    Fit a state-value function via gradient-boosted fitted value iteration (FVI).

    This routine implements fitted value iteration using LightGBM regression trees as the
    function approximator. Given transitions
    (S_t, Y_t, S_{t+1}, is_terminal_outcome_t), it repeatedly:

      1. Forms a temporal-difference (TD) regression target
         Y_t + discount_factor * V(S_{t+1}) * (1 - is_terminal_outcome_t),
      2. Fits (or boosts) a LightGBM model for V(s) to minimize this TD loss,
      3. Monitors a held-out validation TD risk for early stopping.

    The result is a value function approximator V̂(s) for a fixed policy, trained
    purely from batch transitions (offline RL).

    Parameters
    ----------
    S : array-like of shape (n_samples, n_features_S) or (n_samples,)
        Current-state features S_t. Will be reshaped to (n_samples, -1) if needed.

    S_next : array-like of shape (n_samples, n_features_S) or (n_samples,)
        Next-state features S_{t+1}, aligned with S.

    rewards : array-like of shape (n_samples,)
        One-step rewards R_t (or possibly discounted returns, depending on how you
        construct them). This function treats them as immediate rewards in the
        TD target.

    discount_factor : float
        Discount factor γ ∈ [0, 1). Controls the contribution of future value
        V(S_{t+1}) in the TD target.

    is_terminal_outcome : array-like of shape (n_samples,), optional
        Boolean or {0,1} indicator for whether the transition ends in a terminal
        outcome. For terminal transitions (is_terminal_outcome = 1), the TD target
        is Y_t and no future value is added. If None, all transitions are treated
        as non-terminal. Defaults to None.

    weights : array-like of shape (n_samples,), optional
        Sample weights for the TD regression loss. These can encode importance
        sampling weights, state-relevance weights, or per-transition reliability.
        If None, all samples receive equal weight. Defaults to None.

    categorical_feature : list of str or list of int, optional
        Features in S that should be treated as categorical by LightGBM. Passed
        through to the underlying LightGBM Dataset. Defaults to None.

    lgb_params : dict, optional
        LightGBM hyperparameters for the base learner (e.g. num_leaves,
        min_data_in_leaf, learning_rate, etc.). These override the internal
        defaults. The 'objective' field is internally overwritten by a custom
        TD objective at each outer iteration. Defaults to None.

    init_model : lightgbm.Booster, optional
        Initial LightGBM model for warm-starting the value function. If provided,
        value estimates V and V' are initialized using this model and further
        boosted. If None, V is initialized to zero. Defaults to None.

    init_score : array-like of shape (n_samples,), optional
        Optional per-sample offset (baseline) passed to LightGBM as init_score.
        This can be used to center the TD residuals around a known baseline
        value function. Defaults to None.

    test_size : float, optional
        Fraction of data to reserve as a validation set when data_dict_test is
        not provided and train_indices is None. Must be in (0, 1). Defaults to 0.2.

    train_indices : array-like, optional
        Explicit indices of samples to use for training. The complement of these
        indices is used for validation. If None, a random train/validation split
        is created using test_size and seed. Defaults to None.

    data_dict_test : dict, optional
        If provided, an explicit validation dataset is used instead of splitting
        S, S_next, Y. Expected keys:
            "S", "S_next", "Y", "weights", "is_terminal_outcome", "init_score".
        When this is non-None, refit_on_all_data is ignored. Defaults to None.

    refit_on_all_data : bool, optional
        If True and data_dict_test is None, after the outer boosting loop finishes,
        a final model is refit on the full dataset (train + validation) while
        following the same iterative TD procedure. The returned 'model' then
        corresponds to this full-data refit. If False, the model trained only
        on the training split is returned. Defaults to False.

    boost : bool, optional
        If True (default), the algorithm performs genuine gradient boosting:
        at each outer iteration it adds a single new tree (one boosting step)
        using num_iterations = 1 and manually updates predictions. If False,
        it refits a full LightGBM model at each outer iteration using
        num_iterations_outer steps, which can be much more expensive.

    fit_control : dict, optional
        Dictionary controlling the outer FVI loop and early stopping logic.
        The following keys are recognized (with internal defaults):

            - "early_stopping_rounds" : int
                  Number of consecutive non-improving iterations allowed before
                  triggering early stopping (in terms of TD risk or RMSE).
                  Default is 10 when boost is True, 1 otherwise.

            - "early_stopping_min_delta" : float
                  Minimal required decrease in validation criterion per iteration
                  (before scaling by the learning rate). Smaller values make
                  stopping more conservative. Default is 1e-6.

            - "early_stopping_min_learning_rate" : float
                  Minimum learning rate threshold; if the learning rate is
                  repeatedly halved below this value, training stops.
                  Default is min(1e-5, learning_rate / 10000).

            - "update_outcome_rounds" : int
                  Frequency (in outer iterations) with which the TD objective
                  is recomputed using the current V(S_{t+1}). Default is 1
                  (i.e., every iteration).

            - "num_iterations_outer" : int
                  Maximum number of outer FVI iterations when boost=False.
                  Ignored when boost=True, where num_iterations is taken from
                  LightGBM's 'num_iterations' parameter. Default is 1000.

            - "do_early_stopping" : bool
                  Whether to apply early stopping logic at all. If False, the
                  loop runs for the full number of outer iterations. Default True.

            - "stopping_criteria" : {"risk", "rmse"}
                  Criterion used to decide whether an iteration is improving:
                  * "risk": use TD-based validation risk.
                  * "rmse": use RMSE between successive value predictions on
                    the training set.
                  Default is "risk".

        Any subset of these keys can be provided; unspecified keys use defaults.

    seed : int, optional
        Random seed controlling the train/validation split when train_indices
        is None, and passed to LightGBM as 'seed'. Defaults to 42.

    verbose : bool, optional
        If True, periodically prints validation TD risk, normalized change in
        risk, and normalized RMSE across iterations, as well as messages about
        early stopping and learning-rate reductions. Defaults to False.

    Returns
    -------
    dict
        A dictionary with the following entries:

            - "model" : lightgbm.Booster
                  The final value function model to use at test time. If
                  refit_on_all_data is True (and data_dict_test is None), this
                  is the model trained on the full dataset; otherwise, it is the
                  model trained on the training split.

            - "model_train" : lightgbm.Booster
                  The model trained on the training subset only. This is useful
                  for diagnostics or if you want to separate validation-time
                  behavior from a final refit.

            - "train_indices" : np.ndarray or None
                  Indices used for training. If data_dict_test was provided,
                  this is whatever train_indices was passed in (possibly None).

    Notes
    -----
    - This implementation assumes an episodic setting where Y contains
      one-step rewards and is_terminal_outcome marks the end of an episode.
      For terminal transitions, no future value is added to the TD target.

    - All TD objectives and risks are implemented via helper functions
      (create_td_dataset, create_td_objective, risk_fitted_value_iteration)
      from LongTermMarkov.utils, which encapsulate the Bellman error
      computation.

    - When boost=True, we use LightGBM's incremental training interface
      (Booster.update + rollback_one_iter) to add a single tree per outer
      iteration, manually tracking value updates and enabling fine-grained
      early stopping based on RL-style TD risk rather than pure regression
      metrics.
    """
 

    Y = rewards
    
    if not isinstance(Y, np.ndarray):
        Y = np.array(Y).reshape(-1)
    else:
        Y = Y.reshape(-1)

    n = len(Y)

    if not isinstance(S, np.ndarray):
        S = np.array(S).reshape(n, -1)
    else:
        S = S.reshape(n, -1)

    if not isinstance(S_next, np.ndarray):
        S_next = np.array(S_next).reshape(n, -1)
    else:
        S_next = S_next.reshape(n, -1)


    # Set default LightGBM parameters
    params = {
        'objective': "regression",
        'verbosity': -1,
        'bagging_fraction': 1,
        'bagging_freq': 1,
        'seed': seed,
        'max_depth': -1,
        'learning_rate': 0.1,
        'num_iterations': 1000,
        'num_threads': 1,
        'num_leaves': 32,
        'min_data_in_leaf': 100,
        'interaction_constraints': None,
        'lambda_l1': 0.0,
        'lambda_l2': 0.0,
        'min_sum_hessian_in_leaf': 1e-3
    }

    if lgb_params:
        params.update(lgb_params)
    learning_rate = params['learning_rate']

    # Default fit control parameters
    fit_control_default = {
        'early_stopping_rounds': 10 if boost else 1, 
        'early_stopping_min_delta': 1e-6, 
        'early_stopping_min_learning_rate': min(1e-5, learning_rate / 10000),
        'update_outcome_rounds': 1,
        'num_iterations_outer': 1000,
        'do_early_stopping': True,
        'stopping_criteria': 'risk'
    }


    if fit_control:
        fit_control_default.update(fit_control)

    if boost:
        num_iterations = params['num_iterations']
        params['num_iterations'] = 1
    else:
        num_iterations = fit_control_default['num_iterations_outer']

    
    early_stopping_rounds = fit_control_default['early_stopping_rounds']
    early_stopping_min_delta = fit_control_default['early_stopping_min_delta']
    early_stopping_min_learning_rate = fit_control_default['early_stopping_min_learning_rate']
    update_outcome_rounds = fit_control_default['update_outcome_rounds']
    do_early_stopping =  fit_control_default['do_early_stopping']
    stopping_criteria = fit_control_default['stopping_criteria']

    # If no weights or terminal state indicators are provided, set default values
    weights = np.ones_like(Y) if weights is None else weights
    is_terminal_outcome = np.zeros_like(Y) if is_terminal_outcome is None else is_terminal_outcome

 
    # Split data into training and validation sets
    if data_dict_test is not None:
        S_train, S_val = S, data_dict_test.get("S")
        S_next_train, S_next_val = S_next, data_dict_test.get("S_next")
        Y_train, Y_val = Y, data_dict_test.get("Y")
        weights_train, weights_val = weights, data_dict_test.get("weights")
        is_terminal_outcome_train, is_terminal_outcome_val = is_terminal_outcome, data_dict_test.get("is_terminal_outcome")  
        init_score_train, init_score_val = init_score, data_dict_test.get("init_score")
        refit_on_all_data = False # not done if data_dict_test is provided
    else:
        indices = np.arange(len(S))
        if train_indices is None:
            train_indices, val_indices = train_test_split(indices, test_size=test_size, random_state=seed)
        else:
            val_indices = np.setdiff1d(indices, train_indices)

        S_train, S_val = S[train_indices], S[val_indices]
        S_next_train, S_next_val = S_next[train_indices], S_next[val_indices]
        Y_train, Y_val = Y[train_indices], Y[val_indices]
        weights_train, weights_val = weights[train_indices], weights[val_indices]
        is_terminal_outcome_train, is_terminal_outcome_val = is_terminal_outcome[train_indices], is_terminal_outcome[val_indices]
        init_score_train = init_score[train_indices] if init_score is not None else None
        init_score_val = init_score[val_indices] if init_score is not None else None

  
    # Initialize model
    if init_model is not None:
        init_model = lgb.Booster(model_str=init_model.model_to_string())
        init_model_all = lgb.Booster(model_str=init_model.model_to_string())
    else:
        init_model_all = None
    
    current_model = init_model
    current_model_all = init_model_all
    previous_model = current_model
    

    patience_counter = 0
    data_train = None
    data_train_all = None
    force_update = True
    current_validation_risk = np.inf
    initial_learning_rate = learning_rate
    loss_delta = np.inf
    adjusted_early_stopping_min_delta = 0
    if init_model is None:
        V_cur_val = np.zeros_like(Y_val)
        V_cur_train = np.zeros_like(Y_train)
        V_next_cur_val = np.zeros_like(Y_val)
        V_next_cur_train = np.zeros_like(Y_train)
        V_next_cur = np.zeros_like(Y)
    else:
        V_cur_val = init_model.predict(S_val)
        V_cur_train = init_model.predict(S_train)
        V_next_cur_val = init_model.predict(S_next_val)
        V_next_cur_train = init_model.predict(S_next_train)
        V_next_cur = init_model.predict(S_next)
        
    params_all = copy.deepcopy(params)
 

 
    # Main iteration loop
    data_train = create_td_dataset(S_train, S_next_train, Y_train, V_next=V_next_cur_train, discount_factor=discount_factor, is_terminal_outcome=is_terminal_outcome_train, weights=weights_train, data_train=None, categorical_feature=categorical_feature, init_score=init_score_train)

    data_train_all = create_td_dataset(S, S_next, Y, V_next=V_next_cur, discount_factor=discount_factor, is_terminal_outcome=is_terminal_outcome, weights=weights, data_train=None, categorical_feature=categorical_feature, init_score=init_score)
    
    boost_iteration = 0 if init_model is None else init_model.num_trees()
    tolerance = 1e-8 
    mu = mu2 = 0

    # flags
    flag_first_loss_increase = False
    flag_first_calibration_increase = False

 
    for iteration in tqdm(range(num_iterations)):

        if current_model is not None and iteration < 10 and not np.allclose(current_model.predict(S_val), V_cur_val, atol=tolerance):
            raise ValueError("The updated values do not match the predicted values within the specified tolerance.")
        if current_model is not None and iteration < 10 and not np.allclose(current_model.predict(S_next_val), V_next_cur_val, atol=tolerance):
            raise ValueError("The updated values do not match the predicted values within the specified tolerance.")

        # Compute previous V-risk
        current_validation_risk = risk_fitted_value_iteration(V_cur_val, V_next_cur_val, Y_val, discount_factor=discount_factor, is_terminal_outcome=is_terminal_outcome_val, weights=weights_val, init_score=init_score_val)
        
        # Update outcome
        if force_update or iteration % update_outcome_rounds == 0:
            # compute new objectivee
            updated_objective = create_td_objective(Y_train, V_next = V_next_cur_train, discount_factor = discount_factor, is_terminal_outcome=is_terminal_outcome_train, init_score = init_score_train)
            params['objective'] = updated_objective


        
        # Train model on training split
        
        with utils.suppress_lightgbm_output():
            previous_model = current_model
            if boost:
                if iteration == 0 or current_model is None:
                    # Initial training or forced retraining
                    current_model = lgb.train(params, data_train, init_model=current_model, keep_training_booster=True)
                else:
                    # Update the existing model
                    current_model.reset_parameter({'learning_rate': learning_rate})
                    current_model.reset_parameter({'objective': None})
                    current_model.update(fobj = updated_objective)
            else:
                current_model = lgb.train(params, data_train, init_model=None, keep_training_booster=True)

        # manually update boosted predictions for speed
        if boost:
            V_updated_val = V_cur_val + current_model.predict(S_val, start_iteration = boost_iteration )
            V_updated_train = V_cur_train + current_model.predict(S_train, start_iteration = boost_iteration )
            V_next_updated_val = V_next_cur_val + current_model.predict(S_next_val, start_iteration = boost_iteration)
            V_next_updated_train = V_next_cur_train + current_model.predict(S_next_train, start_iteration = boost_iteration)
        else:
            V_updated_val = current_model.predict(S_val)
            V_next_updated_val = current_model.predict(S_next_val)
            V_next_updated_train = current_model.predict(S_next_train)
            V_updated_train = current_model.predict(S_train )

         
    
        
        # check that custom boosted predictions match full predictions
        if iteration < 10 and not np.allclose(V_next_updated_train, current_model.predict(S_next_train), atol=tolerance):
            raise ValueError("The updated values do not match the predicted values within the specified tolerance.")
        if  iteration < 10 and not np.allclose(current_model.predict(S_val), V_updated_val, atol=tolerance):
            raise ValueError("The updated values do not match the predicted values within the specified tolerance.")

        

     
        
  
        validation_risk = risk_fitted_value_iteration(V_updated_val, V_next_cur_val, Y_val, discount_factor, is_terminal_outcome=is_terminal_outcome_val, weights=weights_val, init_score=init_score_val)

        rmse_train = np.sqrt(np.average((V_updated_train - V_cur_train)**2, weights = weights_train))

         


        # Check for early stopping adjusting for learning rate
        adjusted_early_stopping_min_delta = early_stopping_min_delta * learning_rate
        loss_delta = np.inf if np.isinf(current_validation_risk) else (current_validation_risk - validation_risk)
        if loss_delta <= 0:
            flag_first_loss_increase = True
        

        if stopping_criteria == "rmse":
            continue_training = (not flag_first_loss_increase) or (rmse_train >= adjusted_early_stopping_min_delta)
        else:
            continue_training =  validation_risk <= current_validation_risk  

        

         

        if verbose and iteration % np.round(num_iterations / 100) == 0:
            print(f"Iteration {iteration + 1}, Validation risk: {current_validation_risk}. \n"
                  f"Normalized decrease in validation risk: {loss_delta / learning_rate}. "
                  f"Normalized decrease in rmse: {rmse_train / learning_rate}.")


            
        # Update model if improvement otherwise rollback one iteration
        if not do_early_stopping or continue_training:
             
            V_cur_val = V_updated_val
            V_next_cur_val = V_next_updated_val
            V_next_cur_train = V_next_updated_train
            V_cur_train = V_updated_train
            # update model using all data
            if refit_on_all_data:
                if force_update or iteration % update_outcome_rounds == 0:
                    # update objective
                    updated_objective_all = create_td_objective(Y, V_next_cur, discount_factor, is_terminal_outcome=is_terminal_outcome, init_score = init_score)
                    params_all['objective'] = updated_objective_all
                    # update model. The functions lgb.train and lgb.update should do the same thing with the latter being fast
                # get updated predictions from all data
                if boost:
                    if iteration == 0 or current_model_all is None:
                        with utils.suppress_lightgbm_output():
                            current_model_all = lgb.train(params_all, data_train_all, init_model=current_model_all, keep_training_booster=True)
                    else:
                        current_model_all.reset_parameter({'learning_rate': learning_rate})
                        current_model_all.reset_parameter({'objective': None})
                        current_model_all.update(fobj = updated_objective_all)
                    V_next_cur = V_next_cur + current_model_all.predict(S_next, start_iteration = boost_iteration)                    
                else:
                    with utils.suppress_lightgbm_output():
                        current_model_all = lgb.train(params_all, data_train_all, init_model=None, keep_training_booster=True)
                        V_next_cur = current_model_all.predict(S_next)
                if iteration < 10 and not np.allclose(V_next_cur, current_model_all.predict(S_next), atol=tolerance):
                    raise ValueError("The updated values do not match the predicted values within the specified tolerance.")
            # update iteration
            boost_iteration += 1
             
        else:
            # Force update outcome if loss increased
             
            if boost: 
                current_model.rollback_one_iter()
            else:
                current_model = previous_model

        # Early stopping
        if not do_early_stopping or continue_training:
            patience_counter = 0
            force_update = False
        else:
            patience_counter += 1
            # Adjust learning rate  
            if boost: learning_rate = 0.5 * learning_rate
            force_update = True  # Force update outcome if loss increased
            params['learning_rate'] = learning_rate
            params_all['learning_rate'] = learning_rate
            if verbose:
                print(f"Loss increased. The new learning rate is: {learning_rate}")

        # Check if early stopping criteria are met
        if iteration >= min(100, 1 / initial_learning_rate):
            if patience_counter >= early_stopping_rounds or learning_rate < early_stopping_min_learning_rate:
                if verbose:
                    print(f"Early stopping at iteration {iteration + 1} with relative gain {loss_delta:.4f}")
                break
    print("done")
    # Choose the final model based on whether refitting on full data is requested
    if refit_on_all_data:
        # if there is no improvement from fitting then current_model_all is Nobe
        if current_model_all is None:
            current_model_all = current_model
        model_out = current_model_all
        #if current_model_all.num_trees() !=current_model.num_trees():
         #   raise ValueError("Something went wrong... The model trained on all data has a different number of trees then the one trained on training data.")
    else:
        model_out = current_model 
    
    

        
    return {'model': model_out, "model_train": current_model, 'train_indices': train_indices}



 


  
def create_td_dataset(
    S,
    S_next,
    Y,
    discount_factor,
    model=None,
    V_next=None,
    is_terminal_outcome=None,
    weights=None,
    data_train=None,
    categorical_feature=None,
    init_score=None,
):
    """
    Construct a LightGBM Dataset for TD regression.

    This builds TD targets of the form
        Y + discount_factor * V_next * (1 - is_terminal_outcome)
    (or just Y if V_next is None), optionally centered by init_score.

    Parameters
    ----------
    S : array-like of shape (n_samples, n_features) or (n_samples,)
        Current-state features.

    S_next : array-like of shape (n_samples, n_features) or (n_samples,)
        Next-state features. Used only if V_next is None and `model` is provided.

    Y : array-like of shape (n_samples,)
        One-step rewards.

    discount_factor : float
        Discount factor γ ∈ [0, 1).

    model : lightgbm.Booster or None, optional
        Value-function model used to compute V_next = V(S_next) when V_next
        is not provided explicitly.

    V_next : array-like of shape (n_samples,), optional
        Precomputed value estimates V(S_next). If None and `model` is not None,
        they are computed as model.predict(S_next).

    is_terminal_outcome : array-like of shape (n_samples,), optional
        Boolean or {0,1} indicator for terminal transitions. When provided,
        V_next is multiplied by (1 - is_terminal_outcome), so no future value
        is added for terminal transitions.

    weights : array-like of shape (n_samples,), optional
        Sample weights for the regression loss. If None, all samples are
        equally weighted.

    data_train : lgb.Dataset, optional
        Currently ignored; kept only for API compatibility.

    categorical_feature : list of int or str, optional
        Categorical feature specification passed through to LightGBM.

    init_score : array-like of shape (n_samples,), optional
        Per-sample offset. If provided, the TD targets are shifted by
        `-init_score`.

    Returns
    -------
    lgb.Dataset
        LightGBM dataset with features S and labels equal to the TD targets.
    """
    S = np.atleast_2d(S)
    S_next = np.atleast_2d(S_next)
    Y = np.asarray(Y).reshape(-1)

    if V_next is None and model is not None:
        V_next = model.predict(S_next)

    if V_next is not None:
        V_next = np.asarray(V_next).reshape(-1)
        if is_terminal_outcome is not None:
            V_next = (1 - is_terminal_outcome) * V_next
        td_targets = Y + discount_factor * V_next
    else:
        td_targets = Y

    if init_score is not None:
        init_score = np.asarray(init_score).reshape(-1)
        td_targets = td_targets - init_score

    # Rebuilding the Dataset each time is faster than calling set_label.
    dataset = lgb.Dataset(
        S,
        label=td_targets,
        weight=weights,
        free_raw_data=False,
        categorical_feature=categorical_feature,
        init_score=None,
    )

    return dataset


def create_td_objective(Y, V_next, discount_factor, is_terminal_outcome, init_score=None):
    if V_next is not None:
        V_next = np.asarray(V_next).reshape(-1)
        if is_terminal_outcome is not None:
            V_next = (1 - is_terminal_outcome) * V_next
        td_targets = Y + discount_factor * V_next
    else:
        td_targets = Y

    if init_score is not None:
        init_score = np.asarray(init_score).reshape(-1)
        td_targets = td_targets - init_score

    def custom_obj(preds, train_data):
        nonlocal td_targets
        weights = train_data.get_weight()
        if weights is None:
            weights = np.ones_like(preds)
        grad = weights * (preds - td_targets)
        hess = weights
        return grad, hess

    return custom_obj




 

def risk_fitted_value_iteration(V, V_next, Y, discount_factor, is_terminal_outcome=None, weights=None, init_score=None):
    """
    Compute the empirical risk function for fitted value iteration.

    Parameters:
    V (array-like): The current value estimates.
    V_next (array-like, optional): The next value estimates.
    Y (array-like): The rewards.
    discount_factor (float): The discount factor for future rewards.
    is_terminal_outcome (array-like, optional): Boolean array indicating terminal states.
    weights (array-like, optional): Weights for the samples.
    init_score (array-like, optional): Initial scores for the samples.

    Returns:
    float: The mean squared error between the predicted and target values.
    """
    
    if V is None and V_next is None:
        return np.inf
    
    weights = np.ones_like(Y) if weights is None else weights
    init_score = np.zeros_like(Y) if init_score is None else init_score.reshape(-1)
    V_next = np.zeros_like(V) if V_next is None else V_next
    
    if is_terminal_outcome is not None:
        V_next = (1 - is_terminal_outcome) * V_next
                
    targets = Y + discount_factor * V_next - init_score
    errors = targets - V  
    

    return np.average(errors ** 2, weights=weights)

 
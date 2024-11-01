import csv
import os
import gc
import time
import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import config
import re


def kfold_lightgbm_sklearn(data, categorical_feature=None):
    """
    LightGBM model using stratified KFold for cross-validation.

    Arguments:
        data: Path to the folder where files are saved (string).
        categorical_feature: Specify the categorical features (list, default: None).

    Returns:
        df: DataFrame with processed data.
    """
    df = data[data['TARGET'].notnull()]
    test = data[data['TARGET'].isnull()]
    del_features = ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index', 'level_0']
    predictors = [feat for feat in df.columns if feat not in del_features]
    print("Train shape: {}, test shape: {}".format(df[predictors].shape, test[predictors].shape))


    if not config.STRATIFIED_KFOLD:
        folds = KFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    else:
        folds = StratifiedKFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)

    oof_preds = np.zeros(df.shape[0])
    sub_preds = np.zeros(test.shape[0])
    importance_df = pd.DataFrame()
    auc_df = dict()

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(df[predictors], df['TARGET'])):
        train_x, train_y = df[predictors].iloc[train_idx], df['TARGET'].iloc[train_idx]
        valid_x, valid_y = df[predictors].iloc[valid_idx], df['TARGET'].iloc[valid_idx]

        # Inline cleaning of feature names to remove special JSON characters
        train_x.columns = [re.sub(r'[{}\[\]:,"]', '', col) for col in train_x.columns]
        valid_x.columns = [re.sub(r'[{}\[\]:,"]', '', col) for col in valid_x.columns]

        params = {'random_state': config.RANDOM_SEED, 'nthread': config.NUM_THREADS, 'verbose': 200, 'early_stopping_rounds': config.EARLY_STOPPING}
        clf = LGBMClassifier(**{**params, **config.LIGHTGBM_PARAMS})
        
        if not categorical_feature:
            clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], eval_metric='auc')
        else:
            cleaned_predictors = [re.sub(r'[{}\[\]:,"]', '', col) for col in predictors]
            clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], eval_metric='auc', feature_name=cleaned_predictors, categorical_feature=categorical_feature)

        best_iter = clf.best_iteration_
        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=best_iter)[:, 1]
        sub_preds += clf.predict_proba(test[predictors], num_iteration=best_iter)[:, 1] / folds.n_splits

        # Feature importance
        fold_importance = pd.DataFrame()
        fold_importance["feature"] = predictors
        fold_importance["gain"] = clf.booster_.feature_importance(importance_type='gain')
        fold_importance["split"] = clf.booster_.feature_importance(importance_type='split')
        importance_df = pd.concat([importance_df, fold_importance], axis=0)
        
        # Save metric value for each iteration in train and validation sets
        auc_df['train_{}'.format(n_fold+1)]  = clf.evals_result_['training']['auc']
        auc_df['valid_{}'.format(n_fold + 1)] = clf.evals_result_['valid_1']['auc']

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(df['TARGET'], oof_preds))
    test.loc[:, 'TARGET'] = sub_preds.copy()

    # Average feature importance between folds
    mean_importance = importance_df.groupby('feature').mean().reset_index()
    mean_importance.sort_values(by='gain', ascending=False, inplace=True)
    
    # Save feature importance, test predictions as csv
    if config.GENERATE_SUBMISSION_FILES:
        # Generate oof csv
        oof = pd.DataFrame()
        oof['SK_ID_CURR'] = df['SK_ID_CURR'].copy()
        oof['PREDICTIONS'] = oof_preds.copy()
        oof['TARGET'] = df['TARGET'].copy()
        file_name = 'oof{}.csv'.format(config.SUBMISSION_SUFIX)
        oof.to_csv(os.path.join(config.SUBMISSION_DIRECTORY, file_name), index=False)

        # Submission
        sub_path = os.path.join(config.SUBMISSION_DIRECTORY,'submission{}.csv'.format(config.SUBMISSION_SUFIX))
        test[['SK_ID_CURR', 'TARGET']].to_csv(sub_path, index=False)
        imp_path = os.path.join(config.SUBMISSION_DIRECTORY,'feature_importance{}.csv'.format(config.SUBMISSION_SUFIX))
        mean_importance.to_csv(imp_path, index=False)
    return mean_importance

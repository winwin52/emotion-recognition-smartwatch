#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ensemble Learning for Emotion Recognition using Wearable Sensor Data

This script implements Gradient Boosting, AdaBoost, and LightGBM algorithms
for emotion classification from Samsung Gear 2 smartwatch sensor data.

Based on the paper:
"Gradient Boosting, AdaBoost, and LightGBM for Emotion Recognition: A Comparative Analysis on Wearable Sensor Data"
(ICoICT 2024)

Features:
- Heart rate and movement data (accelerometer + gyroscope)
- 107 statistical features extracted from sliding windows
- Binary (happy vs sad) and Multi-class (happy, neutral, sad) classification

Usage:
    # Binary classification (happy vs sad)
    python ensemble_emotion_classification.py -mo features/features_mo* -mu features/features_mu* -mw features/features_mw*

    # Multi-class classification (happy, neutral, sad)
    python ensemble_emotion_classification.py --neutral -mo features/features_mo* -mu features/features_mu* -mw features/features_mw*

Author: Claude Code (replicating the research paper methodology)
"""

from __future__ import print_function
import argparse
import math
import yaml
import numpy as np
from collections import defaultdict

from sklearn import metrics
from sklearn import preprocessing
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold


SEED = 42
np.random.seed(SEED)


def get_models(n_estimators=100, random_state=42):
    """Create dictionary of classifiers to evaluate."""
    models = [
        ('baseline', 'Baseline', DummyClassifier(strategy='most_frequent')),
        ('logit', 'Logistic Regression', LogisticRegression(max_iter=1000, random_state=random_state)),
        ('rf', 'Random Forest', RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)),
        ('adaboost', 'AdaBoost', AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state)),
        ('gb', 'Gradient Boosting', GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)),
        ('lightgbm', 'LightGBM', None)  # LightGBM is added separately
    ]
    return models


def process_condition(fnames, condition, n_estimators=100, include_neutral=False, random_state=42):
    """
    Process a single experimental condition and return classification results.

    Args:
        fnames: List of feature file paths
        condition: Condition name (mo/mu/mw)
        n_estimators: Number of estimators for ensemble methods
        include_neutral: Whether to include neutral class (3-class vs 2-class)
        random_state: Random seed for reproducibility
    """
    print('=' * 70)
    print('Condition: %s' % condition)
    print('=' * 70)

    results = {
        'condition': condition,
        'labels': [],
        'baseline': defaultdict(list),
        'logit': defaultdict(list),
        'rf': defaultdict(list),
        'adaboost': defaultdict(list),
        'gb': defaultdict(list),
        'lightgbm': defaultdict(list)
    }

    # Try to import LightGBM
    try:
        import lightgbm as lgb
        has_lgb = True
    except ImportError:
        print('Warning: LightGBM not installed. Skipping LightGBM model.')
        has_lgb = False

    folds = 10
    repeats = 10
    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=random_state)

    for fname in fnames:
        print('Classifying: %s' % fname)
        label = fname.split('/')[-1]

        # Load data
        data = np.loadtxt(fname, delimiter=',')

        if not include_neutral:
            # Binary classification: remove neutral class (emotion = 0)
            data = np.delete(data, np.where(data[:, -1] == 0), axis=0)

        print('  Shape: %s, Labels: %s' % (str(data.shape), str(np.unique(data[:, -1]))))

        np.random.shuffle(data)

        x_data = data[:, :-1]
        y_data = data[:, -1]

        # Scale features
        x_data = preprocessing.scale(x_data)

        models = [
            ('baseline', 'Baseline', DummyClassifier(strategy='most_frequent')),
            ('logit', 'Logistic Regression', LogisticRegression(max_iter=1000, random_state=random_state)),
            ('rf', 'Random Forest', RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)),
            ('adaboost', 'AdaBoost', AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state)),
            ('gb', 'Gradient Boosting', GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)),
        ]

        if has_lgb:
            import lightgbm as lgb
            models.append(('lightgbm', 'LightGBM', lgb.LGBMClassifier(n_estimators=n_estimators, random_state=random_state, verbose=-1)))

        results['labels'].append(label)

        for key, name, clf in models:
            scores = {'f1': [], 'acc': [], 'auc': []}

            for train_idx, test_idx in rskf.split(x_data, y_data):
                x_train, x_test = x_data[train_idx], x_data[test_idx]
                y_train, y_test = y_data[train_idx], y_data[test_idx]

                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)
                y_proba = clf.predict_proba(x_test)

                _acc = metrics.accuracy_score(y_test, y_pred)

                if include_neutral:
                    _f1 = metrics.f1_score(y_test, y_pred, average='weighted')
                    if len(np.unique(y_test)) > 1:
                        _auc = metrics.roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                    else:
                        _auc = 0.5
                else:
                    _f1 = metrics.f1_score(y_test, y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'weighted')
                    if len(np.unique(y_test)) > 1:
                        _auc = metrics.roc_auc_score(y_test, y_proba[:, 1])
                    else:
                        _auc = 0.5

                scores['f1'].append(_f1)
                scores['acc'].append(_acc)
                scores['auc'].append(_auc)

            results[key]['f1'].append(np.mean(scores['f1']))
            results[key]['acc'].append(np.mean(scores['acc']))
            results[key]['auc'].append(np.mean(scores['auc']))

            print('    %-25s - Acc: %.4f, F1: %.4f, AUC: %.4f' % (name, np.mean(scores['acc']), np.mean(scores['f1']), np.mean(scores['auc'])))

    return results


def print_summary(results, include_neutral=False):
    """Print summary statistics for all models."""
    classification_type = "Multi-class" if include_neutral else "Binary (Happy vs Sad)"

    print('\n' + '=' * 80)
    print('SUMMARY: %s Classification - %s' % (classification_type, results['condition']))
    print('=' * 80)

    print('\n%-25s %-10s %-10s %-10s' % ('Model', 'AUC', 'F1', 'Accuracy'))
    print('-' * 60)

    model_names = {
        'baseline': 'Baseline',
        'logit': 'Logistic Regression',
        'rf': 'Random Forest',
        'adaboost': 'AdaBoost',
        'gb': 'Gradient Boosting',
        'lightgbm': 'LightGBM'
    }

    for key in ['baseline', 'logit', 'rf', 'adaboost', 'gb', 'lightgbm']:
        if key in results and results[key]['auc']:
            auc = np.mean(results[key]['auc'])
            f1 = np.mean(results[key]['f1'])
            acc = np.mean(results[key]['acc'])
            print('%-25s %-10.4f %-10.4f %-10.4f' % (model_names[key], auc, f1, acc))

    print('-' * 60)


def main():
    """Main function to run ensemble emotion classification."""
    parser = argparse.ArgumentParser(
        description='Ensemble Learning for Emotion Recognition using Wearable Sensor Data'
    )
    parser.add_argument(
        '-mo', metavar='MO', type=str, nargs='+',
        help='Feature files for Movie condition (watch movie then walk)',
        default=[]
    )
    parser.add_argument(
        '-mu', metavar='MU', type=str, nargs='+',
        help='Feature files for Music condition (listen to music then walk)',
        default=[]
    )
    parser.add_argument(
        '-mw', metavar='MW', type=str, nargs='+',
        help='Feature files for Music+Walking condition',
        default=[]
    )
    parser.add_argument(
        '-e', '--estimators', type=int, default=100,
        help='Number of estimators for ensemble methods (default: 100)'
    )
    parser.add_argument(
        '-o', '--output', type=str, default='ensemble_results',
        help='Output file prefix for saving results (default: ensemble_results)'
    )
    parser.add_argument(
        '--neutral', action='store_true',
        help='Include neutral class for multi-class classification'
    )
    parser.add_argument(
        '-s', '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()
    neutral = args.neutral

    all_results = {
        'n_estimators': args.estimators,
        'include_neutral': neutral,
        'random_state': args.seed,
        'conditions': {}
    }

    # Process each condition
    if args.mo:
        results = process_condition(args.mo, 'mo', args.estimators, neutral, args.seed)
        all_results['conditions']['mo'] = results
        print_summary(results, neutral)

    if args.mu:
        results = process_condition(args.mu, 'mu', args.estimators, neutral, args.seed)
        all_results['conditions']['mu'] = results
        print_summary(results, neutral)

    if args.mw:
        results = process_condition(args.mw, 'mw', args.estimators, neutral, args.seed)
        all_results['conditions']['mw'] = results
        print_summary(results, neutral)

    # Save results to YAML
    output_file = '%s_neutral.yaml' % args.output if neutral else '%s.yaml' % args.output
    print('\nSaving results to %s' % output_file)

    # Convert numpy types to Python types for YAML serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    serializable_results = convert_to_serializable(all_results)
    with open(output_file, 'w') as f:
        yaml.dump(serializable_results, f, default_flow_style=False)

    # Print overall best models
    print('\n' + '=' * 80)
    print('BEST PERFORMING MODELS (by Accuracy)')
    print('=' * 80)

    classification_type = "Multi-class" if neutral else "Binary"

    for cond_key, cond_name in [('mo', 'Movie (mo)'), ('mu', 'Music (mu)'), ('mw', 'Music+Walking (mw)')]:
        if cond_key in all_results['conditions']:
            cond_results = all_results['conditions'][cond_key]
            best_model = max(cond_results.keys(), key=lambda k: np.mean(cond_results[k]['acc']) if cond_results[k]['acc'] else 0)
            best_acc = np.mean(best_model['acc']) if isinstance(best_model, dict) else np.mean(cond_results[best_model]['acc'])
            model_name = {
                'baseline': 'Baseline',
                'logit': 'Logistic Regression',
                'rf': 'Random Forest',
                'adaboost': 'AdaBoost',
                'gb': 'Gradient Boosting',
                'lightgbm': 'LightGBM'
            }.get(best_model, best_model)
            print('%s: %s with Accuracy = %.4f' % (cond_name, model_name, best_acc))

    print('\n' + '=' * 80)
    print('CLASSIFICATION COMPLETE')
    print('=' * 80)


if __name__ == '__main__':
    main()

"""
This module provides functionality to evaluate model performance and
fine tune threshold values to use
"""

import pandas as pd
import numpy as np
# import ktrain
# from ktrain import text
from ktrain.text.predictor import TextPredictor
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.metrics import roc_curve, auc

# helper function to plot a confusion matrix
def plot_confusion_matrix(
        predictor : TextPredictor,
        data:pd.DataFrame,
        text_column : str,
        label_columns : str,
        title='Confusion matrix of the classifier') -> confusion_matrix:
    """
    Plots a confusion matrix.
    """
    # perform the predictions
    test_descriptions = data[text_column].tolist()
    test_true = data[label_columns].tolist()
    test_pred = predictor.predict(test_descriptions)

    # create the confusion matrix
    labels = predictor.get_classes()
    cm = confusion_matrix(test_true, test_pred)

    # plot the confusion matrix
    sns.set(rc={'figure.figsize':(21.7,18.27)})
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Greens', fmt='g')

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(title)
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

    plt.yticks(rotation=0)
    plt.xticks(rotation=90)

def print_classification_report(predictor : TextPredictor,
                                data:pd.DataFrame,
                                text_column : str,
                                label_columns : str) -> None:
    """
    Prints classification report for given data
    """
    # extract the required columns
    test_descriptions = data[text_column].tolist()
    test_true = data[label_columns].tolist()

    # perform the predictions
    test_pred = predictor.predict(test_descriptions)

    # print the classification report
    print('Classification report:')
    print(classification_report(test_true, test_pred))
    print('')

def predict_probabilities(predictor : TextPredictor,
                          data:pd.DataFrame,
                          text_column : str,
                          label_columns : str) -> pd.DataFrame:
    """
    Predicts probabilities for a given dataset.
    """
    # perform the predictions
    test_descriptions = data[text_column].tolist()
    test_true = data[label_columns].tolist()
    test_pred_prob = predictor.predict(test_descriptions, return_proba=True)

    # create the output dataset
    classes = predictor.get_classes()

    df_test_pred = pd.DataFrame(test_pred_prob) \
        .assign(pred_conf=lambda df: df.max(axis=1)) \
        .assign(pred_id=lambda df: df.idxmax(axis=1)) \
        .assign(true_name=test_true)

    df_test_pred['pred_name'] = df_test_pred.apply(lambda row : classes[int(row.pred_id)], axis=1)

    return df_test_pred

def calculate_class_metrics(
        predicted : pd.DataFrame, selected_class : str) -> pd.DataFrame:
    """
    Calculate precision and recall for the given class for difference
    confidence levels.
    """
    # find false positive rate for Data top-up
    detection_rates = []
    step_size = 0.01

    # get the total number of true positives for the selected class
    total_tp = predicted \
        .query('true_name == @selected_class') \
        .shape[0]

    for current_conf in np.arange(0, 1, step_size):
        current_rate = {'confidence': current_conf}

        # get the number of true positives
        current_rate['tp'] = predicted \
            .query('    pred_name == @selected_class ' + \
                'and true_name == @selected_class ' + \
                'and pred_conf >= @current_conf') \
            .shape[0]

        # get the number of false positives
        current_rate['fp'] = predicted \
            .query('    pred_name == @selected_class ' + \
                'and true_name != @selected_class ' + \
                'and pred_conf >= @current_conf') \
            .shape[0]

        # calculate the precision
        if (current_rate['tp'] + current_rate['fp']) > 0:
            current_rate['precision'] = \
                current_rate['tp'] / (current_rate['tp'] + current_rate['fp'])

        # calculate the recall
        current_rate['recall'] = current_rate['tp'] / total_tp

        # add the results to the output
        detection_rates.append(current_rate)

    return pd.DataFrame(detection_rates)

def plot_class_metrics(detection_rates : pd.DataFrame, 
                       class_name : str = None) -> None:
    """
    Plots precision and recall for different confidence levels.
    """
    sns.set(rc = {'figure.figsize':(15,8)})

    ax = sns.lineplot(
        data=detection_rates,
        y='precision',
        x='confidence',
        label='precision')

    ax2 = ax.twinx()
    sns.lineplot(
        data=detection_rates,
        y='recall',
        x='confidence',
        color='orange',
        ax=ax2,
        label='recall')

    ax.set_xlabel('Confidence level')

    title = 'Precision and recall at different confidence levels'
    if class_name is not None:
        title = f'{class_name} :: {title}'

    ax.set_title(title)
    plt.show()
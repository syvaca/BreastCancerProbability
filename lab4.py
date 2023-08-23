import numpy as np
import pandas as pd
from lab4_utils import feature_names

# Hint: Consider how to utilize np.unique()
def preprocess_data(training_inputs, testing_inputs, training_labels, testing_labels):
    processed_training_inputs, processed_testing_inputs = ([], [])
    processed_training_labels, processed_testing_labels = ([], [])
    # VVVVV YOUR CODE GOES HERE VVVVV $
   
    # pre-process data where missing instances are filled with mode of that feature
    row_names = ['age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'bresast-quad', 'irradiat']
    
    # training inputs
    df = pd.DataFrame(training_inputs, columns = row_names)
    df=df.replace({'?':np.NaN})
    for x in row_names:
        df[x].fillna(df[x].mode()[0], inplace = True)
    processed_training_inputs = df.to_numpy().tolist()    

    #testing inputs
    df2 = pd.DataFrame(testing_inputs, columns = row_names)
    df2=df2.replace({'?':np.NaN})
    for x in row_names:
        df2[x].fillna(df2[x].mode()[0], inplace = True)
    processed_testing_inputs = df2.to_numpy().tolist()

    # training labels
    df3 = pd.DataFrame(training_labels, columns = ["recurrence"])
    df3=df3.replace({'?':np.NaN})
    df3["recurrence"].fillna(df3["recurrence"].mode()[0], inplace = True)
    processed_training_labels = df3.to_numpy().flatten().tolist()

    #testing labels
    df4 = pd.DataFrame(testing_labels, columns = ["recurrence"])
    df4=df4.replace({'?':np.NaN})
    df4["recurrence"].fillna(df4["recurrence"].mode()[0], inplace = True)
    processed_testing_labels = df4.to_numpy().flatten().tolist()

    # ^^^^^ YOUR CODE GOES HERE ^^^^^ $
    return processed_training_inputs, processed_testing_inputs, processed_training_labels, processed_testing_labels


# Hint: consider how to utilize np.count_nonzero()
def naive_bayes(training_inputs, testing_inputs, training_labels, testing_labels):
    assert len(training_inputs) > 0, f"parameter training_inputs needs to be of length 0 or greater"
    assert len(testing_inputs) > 0, f"parameter testing_inputs needs to be of length 0 or greater"
    assert len(training_labels) > 0, f"parameter training_labels needs to be of length 0 or greater"
    assert len(testing_labels) > 0, f"parameter testing_labels needs to be of length 0 or greater"
    assert len(training_inputs) == len(training_labels), f"training_inputs and training_labels need to be the same length"
    assert len(testing_inputs) == len(testing_labels), f"testing_inputs and testing_labels need to be the same length"
    misclassify_rate = 0
    # VVVVV YOUR CODE GOES HERE VVVVV $
    reccurence = []
    non_reccurence = []

    age = {'10-19': [],'20-29': [], '30-39': [], '40-49': [], '50-59': [], '60-69': [], '70-79': [], '80-89': [], '90-99': []}
    menopause = {'lt40':[], 'ge40':[], 'premeno':[]}
    tumor_size = {'0-4':[], '5-9':[], '10-14':[], '15-19':[], '20-24':[], '25-29':[], '30-34':[], '35-39':[], '40-44':[], '45-49':[], '50-54':[], '55-59':[]}
    inv_nodes = {'0-2':[], '3-5':[], '6-8':[], '9-11':[], '12-14':[], '15-17':[], '18-20':[], '21-23':[], '24-26':[], '27-29':[], '30-32':[], '33-35':[], '36-39':[]}
    node_caps = {'yes':[], 'no':[]}
    deg_malig = {1:[], 2:[], 3:[]}
    breast = {'left':[], 'right':[]}
    breast_quad = {'left_up':[], 'left_low':[], 'right_up':[], 'right_low':[], 'central':[]}
    irradiat = {'yes':[], 'no':[]}

    # create a list of all the dictionaries
    all_dicts = [age, menopause, tumor_size, inv_nodes, node_caps, deg_malig, breast, breast_quad, irradiat]

    # fill the dictionaries with the indices of the training inputs
    for i in range(len(training_inputs)):
        for x in range(9):
            all_dicts[x][training_inputs[i][x]].append(i)

    # P(reccurence|X) = P(X|reccurence)*P(reccurence)
    
    # P(reccurence)
    for i in range(len(training_labels)):
        if training_labels[i] == 'recurrence-events':
            reccurence.append(i)
        else:
            non_reccurence.append(i) 
    prob_recc = (len(reccurence) + 1) / (len(training_inputs) + 2)
    prob_non_recc = (len(non_reccurence) + 1) / (len(training_inputs) + 2)

    # #P(X|reccurence) 
    count = 0
    for i in testing_inputs:
        probabilities = 1
        probabilities1 = 1
        for x in range(9):
            probabilities *= (len(list(set(all_dicts[x][i[x]])&set(reccurence))) + 1) / (len(reccurence) + len(age))
            probabilities1 *= (len(list(set(all_dicts[x][i[x]])&set(non_reccurence))) + 1) / (len(non_reccurence) + len(age))    
        rec1 = probabilities * prob_recc
        rec2 = probabilities1 * prob_non_recc
        if rec1 > rec2:
            if testing_labels[count] == 'no-recurrence-events':
                misclassify_rate += 1
        else:
            if testing_labels[count] == 'recurrence-events':
                misclassify_rate += 1
        count += 1
    
    misclassify_rate /= len(testing_labels)
    
    # ^^^^^ YOUR CODE GOES HERE ^^^^^ $
    return misclassify_rate


# Hint: reuse naive_bayes to compute the misclassification rate for each fold.
def cross_validation(training_inputs, testing_inputs, training_labels, testing_labels):
    data = np.concatenate((training_inputs, testing_inputs))
    label = np.concatenate((training_labels, testing_labels))
    average_rate = 0
    # VVVVV YOUR CODE GOES HERE VVVVV $

    # VVVVV YOUR CODE GOES HERE VVVVV $
    return average_rate
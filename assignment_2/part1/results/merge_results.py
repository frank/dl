import os
import pickle

input_lengths = []
accuracies = []

files = os.listdir('lstm/')
for file in files:
    with open('lstm/' + file, 'rb') as handle:
        data = pickle.load(handle)
        input_lengths.append(data['input_length'])
        accuracies.append(data['final_accuracy'])

with open('results_' + str(input_lengths[0]) + '_' + str(input_lengths[-1]) + '.pkl', 'wb') as handle:
    pickle.dump({'input_lengths': input_lengths, 'accuracies': accuracies}, handle)

from train_1805116 import *
import pickle

from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def test(network, final_layer, test_loader):
    total_correct_predictions = 0
    test_loss = 0.0
    predictions = []
    true_targets = []

    # Collect all test data and labels into lists
    for inputs, targets in test_loader:
        inputs = inputs.view(-1, 28 * 28)
        inputs = inputs / 255.0
        inputs = inputs.numpy()
        inputs = inputs.T
        targets = targets.numpy() - 1
        true_targets=targets
        targets = np.eye(26)[targets]

        output = predict(network, inputs, False)
        output = final_layer.forward(output, targets)
        test_loss += final_layer.loss()

        # predictions.extend(np.argmax(output, axis=1))
        predictions = np.argmax(output, axis=0)
        correct_predictions = np.sum(predictions == true_targets)
        total_correct_predictions += correct_predictions

    # true_targets = np.array(true_targets)
    # predictions = np.array(predictions)

    test_loss /= len(test_loader)
    test_accuracy = total_correct_predictions / len(test_loader.dataset)
    test_f1 = f1_score(true_targets, predictions, average='macro')
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Test F1: {test_f1}")
    #confusion matrix
    conf_matrix = confusion_matrix(true_targets, predictions)
    plt.figure(figsize=(20, 16))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.title('Confusion Matrix')
    plt.show()

    




# independent_test_dataset = ds.EMNIST(root='./data', split='letters',
#                              train=False,
#                              transform=transforms.ToTensor(),
#                              download=True)
with open('ids7.pickle', 'rb') as ids7:
  independent_test_dataset = pickle.load(ids7)

# DataLoader for independent test set
test_loader = DataLoader(independent_test_dataset, batch_size=len(independent_test_dataset))

# Load the model
L1=Dense(28*28, 1024)
L2=ReLU()
L3=Dropout(0.25)
L4=Dense(1024,26)


network = [
    L1,
    L2,
    L3,
    L4
]

L5=softMax_crossEntropy()

# Load the weights and biases from the pickle file
with open('model_1805116.pickle', 'rb') as f:
    data = pickle.load(f)
    weights = data['weights']
    biases = data['biases']

# Update weights and biases for the Dense layers in your network
# dense_layers = [layer for layer in network if isinstance(layer, Dense)]
# for i, layer in enumerate(dense_layers):
#     layer.weights = weights[i]
#     layer.bias = biases[i]

network[0].weights = weights[0]
network[0].bias = biases[0]
network[3].weights = weights[1]
network[3].bias = biases[1]


# Evaluate the model on the entire independent test set
test(network, L5, test_loader)



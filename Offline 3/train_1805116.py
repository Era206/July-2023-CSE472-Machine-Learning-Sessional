import numpy as np
import torch
from sklearn.metrics import f1_score
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import torchvision.datasets as ds
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input, train_flag):
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate, train_flag):
        # TODO: update parameters and return input gradient
        pass

class Dense(Layer):
    def __init__(self, input_size, output_size):
        # Initialize weights and biases
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2. / (output_size + input_size))
        self.bias = np.random.randn(output_size, 1) * np.sqrt(2. / (output_size + input_size))
        # Initialize Adam optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.bias)
        self.v_b = np.zeros_like(self.bias)
        self.t = 0

    def forward(self, input, train_flag=True):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate, train_flag=True):
        self.t += 1
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)

        # Adam optimizer for weights
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * weights_gradient
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (weights_gradient ** 2)
        m_w_hat = self.m_w / (1 - self.beta1 ** self.t)
        v_w_hat = self.v_w / (1 - self.beta2 ** self.t)
        self.weights -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)

        # Adam optimizer for biases
        temp_grad = np.sum(output_gradient, axis=1, keepdims=True)
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * temp_grad
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (temp_grad ** 2)
        m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
        v_b_hat = self.v_b / (1 - self.beta2 ** self.t)
        self.bias -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        return input_gradient
    
class Dropout(Layer):
    def __init__(self, p):
        self.p = p
        self.mask = None

    def forward(self, input, train_flag=True):
        # self.mask = np.random.binomial(1, self.p, size=input.shape) / self.p
        # return input * self.mask
        if(train_flag):
            self.mask = np.random.binomial(1, self.p, size=input.shape) / self.p
            return input * self.mask
        else:
            return input
    def backward(self, output_gradient, learning_rate, train_flag=True):
        # return output_gradient * self.mask
        if(train_flag):
            return output_gradient * self.mask
        else:
            return output_gradient
        

class softMax_crossEntropy():
    def __init__(self):
        self.input = None
        self.target = None
        self.output = None

    def forward(self, input, target):
        self.input = input
        self.target = target
        corrected_input = input - np.max(input, axis=0)
        self.output = np.exp(corrected_input) / np.sum(np.exp(corrected_input), axis=0)
        return self.output

    def backward(self):
        return self.output - self.target.T
    
    def loss(self):
        # print(self.output.shape)
        # print(self.target.shape[0])

        return -np.sum(self.target * np.log(self.output.T + 1e-8))/self.target.shape[0]
    

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input, train_flag=True):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate, train_flag=True):
        return np.multiply(output_gradient, self.activation_prime(self.input))
    

class ReLU(Activation):
    def relu(self, x):
        return np.maximum(0, x)

    def relu_prime(self, x):
        return np.where(x > 0, 1, 0)

    def __init__(self):
        super().__init__(self.relu, self.relu_prime)


# training function
def predict(network, input, train_flag=True):
    output = input
    for layer in network:
        output = layer.forward(output, train_flag)
    return output

def train(network, final_layer, train_loader, val_loader, epochs=1000, learning_rate=0.01, verbose=True):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []
    val_predictions=None
    best_f1=0.0
    for e in range(epochs):
        total_correct_predictions = 0
        train_loss = 0.0
        val_loss = 0.0
        
        
        for inputs, targets in train_loader:
            # inputs = inputs.T  # Transpose inputs if required
            # targets = targets.T  # Transpose targets if required
            #input is now a 28*28 image, make it 1d array
            inputs = inputs.view(-1, 28*28)
            inputs = inputs/255.0
            #covert into numpy array & transpose
            inputs = inputs.numpy()
            inputs = inputs.T
            targets = targets.numpy()
            targets = targets-1
            true_targets = targets
            #doing one hot encoding to targets
            targets = np.eye(26)[targets]

            
            output = predict(network, inputs, True)
            output = final_layer.forward(output, targets)
            train_loss += final_layer.loss()
            
            grad = final_layer.backward()
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate, True)

            predictions = np.argmax(output, axis=0)
            correct_predictions = np.sum(predictions == true_targets)
            total_correct_predictions += correct_predictions

            #accuracy calculation
            accuracy = 0
        
        train_loss /= len(train_loader)
        train_accuracy = total_correct_predictions/len(train_loader.dataset)
        print(f"{e + 1}/{epochs}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}")

        total_correct_validation_predictions = 0

        # Validation
        for inputs, targets in val_loader:
            #input is now a 28*28 image, make it 1d array
            inputs = inputs.view(-1, 28*28)
            inputs = inputs/255.0
            #covert into numpy array & transpose
            inputs = inputs.numpy()
            inputs = inputs.T
            targets = targets.numpy()
            targets = targets-1
            true_targets = targets
            #doing one hot encoding to targets
            targets = np.eye(26)[targets]

            output = predict(network, inputs, False)
            output = final_layer.forward(output, targets)
            val_loss += final_layer.loss()

            predictions = np.argmax(output, axis=0)
            correct_predictions = np.sum(predictions == true_targets)
            total_correct_validation_predictions += correct_predictions
            #validation f1
            val_f1 = f1_score(true_targets, predictions, average='macro')
            if(val_f1>best_f1):
                best_f1=val_f1
                #save model with pickle
                with open('best_model.pkl', 'wb') as f:
                    pickle.dump(network, f)
                print("model saved")
                # print("best f1 score: ",best_f1)
                # print("val f1 score: ",val_f1)


            print(f"{e + 1}/{epochs}, Validation F1: {val_f1}")
            val_f1_scores.append(val_f1)
            val_predictions = predictions

        val_loss /= len(val_loader)
        val_accuracy = total_correct_validation_predictions/len(val_loader.dataset)
        print(f"{e + 1}/{epochs}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        # val_f1_scores.append(val_f1)


    # Plotting
    # Plotting Train and Validation Losses
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Losses')
    plt.legend()
    plt.show()

    # Plotting Train and Validation Accuracies
    plt.figure(figsize=(8, 6))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracies')
    plt.legend()
    plt.show()

    # Plotting Validation F1 Scores
    plt.figure(figsize=(8, 6))
    plt.plot(val_f1_scores, label='Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    plt.legend()
    plt.show()

    # Confusion Matrix
    conf_matrix = confusion_matrix(true_targets, val_predictions)
    plt.figure(figsize=(20, 16))
    # disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    # disp.plot()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.title('Confusion Matrix')
    plt.show()




train_validation_dataset = ds.EMNIST(root='./data', split='letters',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)


independent_test_dataset = ds.EMNIST(root='./data', split='letters',
                             train=False,
                             transform=transforms.ToTensor())


train_size = int(0.85 * len(train_validation_dataset))
val_size = len(train_validation_dataset) - train_size

train_dataset, val_dataset = random_split(train_validation_dataset, [train_size, val_size])
train_batch_size=1024
#save dataset as csv
with open('train_dataset.csv', 'wb') as f:
    pickle.dump(train_dataset, f)


# DataLoader for training set
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

# DataLoader for validation set
val_loader = DataLoader(val_dataset, batch_size=val_size)
    
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

train(network, softMax_crossEntropy(), train_loader, val_loader, epochs=100, learning_rate=0.01)





# # Load the saved model
# with open('best_model.pkl', 'rb') as f:
#     loaded_model = pickle.load(f)

# # Extract weights and biases from the loaded model
# weights = []
# biases = []
# for layer in loaded_model:
#     if isinstance(layer, Dense):  # Assuming the weights and biases are in Dense layers
#         weights.append(layer.weights)
#         biases.append(layer.bias)

# # Save the extracted weights and biases into another pickle file
# extracted_data = {'weights': weights, 'biases': biases}
# with open('model_1805116.pickle', 'wb') as f:
#     pickle.dump(extracted_data, f)
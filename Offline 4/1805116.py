import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import imageio

# each column is separated by a comma
def load_data(filename):
    data = np.loadtxt(os.path.join(os.path.dirname(__file__), 'dataset', filename), delimiter=',')
    return data


def task1(filename, n):
    data = load_data(filename)
    # print(data)
    # print(data.shape)
    #check datas column count
    if data.shape[1] > 2:
        #row wise mean, and then negate it from each row
        meanData = np.mean(data, axis=0)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i][j] = data[i][j] - meanData[j]
                
        #svd decomposition
        u, s, vh = np.linalg.svd(data, full_matrices=True)
        # print(vh)
        # print(vh.shape)
        # print(u.shape)
        # print(s.shape)
    
        #multiply data with vh.T
        data = np.dot(data, vh.T[:,:n])
        # print(data.shape)
        #plot this data
        plt.scatter(data[:,0], data[:,1])
        
        #save the plot as an image
        plt.savefig(f'{filename.split(".")[0]}/pca.png')
        plt.show()

    elif data.shape[1] == 2:
        plt.scatter(data[:,0], data[:,1])
        # plt.show()
        #save the plot as an image
        plt.savefig(f'{filename.split(".")[0]}/pca.png')
        plt.show()

    else:
        print("Invalid data shape")
    return data


def init(data, k):
    #mean array will have k element from random data
    mean = data[np.random.choice(data.shape[0], k, replace=False)]
    # print (mean)
    #mixing coefficient will have k element and each element will be 1/k
    mixing_coefficients = np.ones(k) / k
    # print (mixing_coefficients)
    #covar matrix will k identity matrix of size data.shape[1]
    covariance = np.array([np.eye(data.shape[1])] * k)
    # print (covariance)
    return mean, covariance, mixing_coefficients

#exxpectation function
def EFunct(data, mean, covariance, mixing_coefficients):
    #for each of k clusters, take the mean, covariance and mixing coefficient, and calculate the probability of each data point belonging to that cluster
    probabilities = np.zeros((len(mixing_coefficients), data.shape[0]))
    for k in range(len(mixing_coefficients)):
        #for total data, calculate the probability of it belonging to each cluster
        mvn=multivariate_normal(mean[k], covariance[k])
        pdf=mvn.pdf(data)
        probabilities[k]=mixing_coefficients[k]*pdf

    #normalize the probabilities
    probabilities = probabilities / np.sum(probabilities, axis=0)
    return probabilities

#maximization function
def MFunct(data, probabilities):
    # Number of clusters
    K = probabilities.shape[0]
    
    # Compute the sum of the probabilities for each cluster
    responsibilities = np.sum(probabilities, axis=1)
    
    # Update mixing coefficients
    updated_mixing_coefficients = responsibilities / data.shape[0]
    
    # Update means
    updated_means = (probabilities @ data) / responsibilities[:, None]
    
    # Update covariances
    updated_covariances = np.zeros((K, data.shape[1], data.shape[1]))
    for k in range(K):
        # Center the data by subtracting the mean
        centered_data = data - updated_means[k]
        # Calculate the outer product and weight by the probabilities, then sum
        cov = (probabilities[k, :, None, None] * centered_data[:, :, None] @ centered_data[:, None, :])
        updated_covariances[k] = cov.sum(axis=0) / responsibilities[k]

    return updated_means, updated_covariances, updated_mixing_coefficients

#log likelihood function
def compute_log_likelihood(data, means, covariances, mixing_coefficients):
    N = data.shape[0]
    K = means.shape[0]
    log_likelihood = 0

    for k in range(K):
        # Create a multivariate normal distribution for each component
        rv = multivariate_normal(means[k], covariances[k])
        # Evaluate the PDF of the component on the data and multiply by the mixing coefficient
        component_density = mixing_coefficients[k]*rv.pdf(data)
        # If this is the first component, initialize the total density
        if k == 0:
            total_density = component_density
        else:
            # Add the component density to the total density
            total_density += component_density

    # Take the logarithm of the total density and sum over all data points
    log_likelihood = np.sum(np.log(total_density))

    return log_likelihood



def plot_gmm(data, means, covariances, probabilities, K, filename):
    # Create a figure
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, K))

    # Assign each data point to the cluster with the highest probability
    cluster_assignments = np.argmax(probabilities, axis=0)

    # Plot data points with colors based on cluster assignment
    for k in range(K):
        cluster_data = data[cluster_assignments == k]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=10, color=colors[k], label=f'Cluster {k+1}')

    # Prepare the grid for contour plots
    x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
    y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])
    x, y = np.mgrid[x_min:x_max:.01, y_min:y_max:.01]
    pos = np.dstack((x, y))

    # Plot the Gaussian components
    for k in range(K):
        # Ensure the mean vector has the correct shape (2,)
        mean_vector = means[k].flatten()[:2]  # Use only the first two elements for 2D space
        
        # Ensure the covariance matrix is 2D and of shape (2, 2)
        covariance_matrix = covariances[k]
        if covariance_matrix.ndim == 1:
            covariance_matrix = np.diag(covariance_matrix[:2])  # Use the first two elements if it's a 1D array
        elif covariance_matrix.shape[0] != 2:
            # Reshape or truncate the covariance matrix to shape (2, 2)
            covariance_matrix = covariance_matrix.reshape(2, 2)

        rv = multivariate_normal(mean_vector, covariance_matrix)
        plt.contour(x, y, rv.pdf(pos), levels=5, colors='r')

    plt.title(f'GMM with K={K}')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.legend()
    plt.grid(True)

    plt.savefig(filename)
    plt.close()

    return imageio.imread(filename)




def task2(dataFile,data, K, iterations, initializations):
  
    # print(data.shape)
    #Choose a range for the number of Gaussian components K, which could be from 3 to 8, do it
    K_range = range(3, 9)
   
    log_likelihoods = np.zeros(len(K_range))
    #take an array for the values of k_range
    
    K_range = np.array(K_range)
   
    gif_images = []

    for i in K_range:
        best_probabilities_k = None  
        best_log_likelihood_k=-np.inf 
        # filenames.clear()    
        for k_init in range(initializations):
            #initialize the parameters
            mean, cov, mixCo_Ef =init(data, i)
            best_log_likelihood = -np.inf
        
            best_probabilities=None
            prev_log_likelihood = -np.inf
            #for each iteration do the following
            for j in range(iterations):
                print(f'K: {i}, Initialization: {k_init}, Iteration: {j}')
                # prev_log_likelihood = best_log_likelihood
                #E-step
                probabilities = EFunct(data, mean, cov, mixCo_Ef)
                #M-step
                mean, cov, mixCo_Ef = MFunct(data, probabilities)
                #calculate the log likelihood
                log_likelihood=compute_log_likelihood(data, mean, cov, mixCo_Ef)

                
                best_log_likelihood = log_likelihood
                best_probabilities = probabilities
                if abs(log_likelihood-prev_log_likelihood) < 0.1:
                    break
                prev_log_likelihood = log_likelihood

                filename = f'gmm_{i}_init_{k_init}_iter_{j}.png'
                plot_gmm(data, mean, cov, probabilities, i, filename)
                gif_images.append(imageio.imread(filename))
                
                os.remove(filename)
                

            if best_log_likelihood > best_log_likelihood_k:
                best_log_likelihood_k = best_log_likelihood
                best_probabilities_k = best_probabilities

            

            last_frame_filename = f'{dataFile.split(".")[0]}/final_image_{i}.png'   
            imageio.imwrite(last_frame_filename, gif_images[-1])
            imageio.mimsave(f'{dataFile.split(".")[0]}/gmm_{i}.gif', gif_images, duration=1.2)  # Adjust duration as needed
            gif_images.clear()  # Clear the list for the next K
            
                
        log_likelihoods[i-3] = best_log_likelihood_k
        
        # # print(best_probabilities_k.shape)
        # # print(np.argmax(best_probabilities_k, axis=0).shape)
        # plt.scatter(data[:,0], data[:,1], c=np.argmax(best_probabilities_k, axis=0))
        # # plt.show()
        # plt.savefig(f'{dataFile.split(".")[0]}/{i}.png')
        # plt.show()
            
    #plot the log likelihoods array against the K_range
    print(log_likelihoods)
    print(K_range)
    
    #save the plot as an image
    plt.plot(K_range, log_likelihoods)
    
    #folder name for the dataset  
    plt.savefig(f'{dataFile.split(".")[0]}/log_likelihood_vs_k.png')
    plt.show()
    
    


def main():
    dataset = "6D_data_points.txt"
    #create folder for the dataset
    if not os.path.exists(f'{dataset.split(".")[0]}'):
        os.mkdir(f'{dataset.split(".")[0]}')
    data = task1(dataset, 2)
    task2(dataset,data, 3, 1000, 5)


if __name__ == "__main__":
    main()
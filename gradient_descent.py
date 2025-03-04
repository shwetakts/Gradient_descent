import numpy as np
import matplotlib.pyplot as plt
import time

def generate_data(x,m1,m2):
    """
    Generates a dataset based on a linear function with added noise.

    Args:
        x (numpy.ndarray): The input values.
        m1 (float): The slope of the line.
        m2 (float): The intercept of the line.

    Returns:
        numpy.ndarray: The generated output values with noise.
    """
    noise=np.random.normal(0,1,len(x))
    return m1*x+m2+noise  

def calc_mse(m1,m2,x,y):
    """
    Calculates the Mean Squared Error (MSE) for a given linear model.

    Args:
        m1 (float): The slope of the line.
        m2 (float): The intercept of the line.
        x (numpy.ndarray): The input values.
        y (numpy.ndarray): The actual output values.

    Returns:
        float: The calculated Mean Squared Error (MSE).
    """
    y_pred=m1*x+m2
    return np.mean((y-y_pred)**2)

def linear_search(m1_values,x,y,m2):
    """
    Finds the best slope (m1) using a brute-force linear search method.

    Args:
        m1_values (numpy.ndarray): Array of possible slope values.
        x (numpy.ndarray): The input values.
        y (numpy.ndarray): The actual output values.
        m2 (float): The intercept of the line.

    Returns:
        float: The best slope (m1) value that minimizes the loss.
    """
    loss=list(map(lambda m1: calc_mse(m1,m2,x,y), m1_values))
    best_i=np.argmin(loss)
    return m1_values[best_i]

def gradient_descent(x, y, m, m2, learning_rate, tolerance,prev_loss, loss, m_values, max_iterations):
    """
    Performs gradient descent optimization recursively to find the optimal slope (m).

    Args:
        x (numpy.ndarray): The input values.
        y (numpy.ndarray): The actual output values.
        m (float): The initial slope value.
        m2 (float): The intercept of the line.
        learning_rate (float): The learning rate for gradient descent.
        tolerance (float): The stopping criteria for convergence.
        prev_loss (float): The previous loss value.
        loss (list): The history of loss values.
        m_values (list): The history of m values.
        max_iterations (int): The maximum number of iterations.

    Returns:
        tuple: (optimal m value, loss history, m value history)
    """
    for i in range(max_iterations):
        y_pred=m*x+m2
        #calculating gradient
        grad=-2*np.mean(x*(y-y_pred))  
        m-=learning_rate*grad
        current_loss=calc_mse(m, m2, x, y)

        #storing the values
        loss.append(current_loss)
        m_values.append(m)

        #early stopping implementation
        if abs(prev_loss-current_loss)<=tolerance:
            print(f"Early stopping at iteration {i}")
            break
        prev_loss=current_loss
    return m, loss, m_values
        


def main():
    """
    Main function to execute the program.
    """
    #data generation
    x=np.linspace(1, 10, 100)
    m1=9
    m2=5
    y=generate_data(x, m1, m2)

    m1_values=np.linspace(-50, 50, 30)

    # Linear Search
    start=time.time()
    best_m1=linear_search(m1_values, x, y, m2)
    end=time.time()
    print(f"Best m1 value from linear search: {best_m1:.5f}")
    print(f"Time taken for Linear Search: {end - start:.5f} seconds")

    # Gradient Descent
    start=time.time()
    m_initial=np.random.randn()
    m,loss,m_values=gradient_descent(x, y, m_initial, m2, 0.0001, 1e-1,float('inf'),[], [],500)
    end=time.time()
    print(f"Estimated m from Gradient Descent: {m:.5f}")
    print(f"Time taken for Gradient Descent: {end - start:.5f} seconds")

    #plot results
    #plotting the datapoints
    plt.scatter(x, y, label='Data Points')
    plt.legend()
    plt.show()

    #plotting the loss function
    plt.figure(figsize=(8, 6))
    plt.plot(m1_values, [calc_mse(m, m2, x, y) for m in m1_values], label="Loss vs m1")
    plt.xlabel('m1')
    plt.ylabel("Loss")
    plt.title("Loss Function")
    plt.legend()
    plt.grid(True)
    plt.show()

    #plotting gradient descent convergence
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(loss)), loss, linestyle='-', color='b', label="Loss Curve")
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Gradient Descent Convergence')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the program
if __name__=="__main__":
    main()

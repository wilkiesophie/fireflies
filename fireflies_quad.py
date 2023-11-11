import numpy as np
import matplotlib.pyplot as plt

def objective_function(x):
    return x**2 + 5*x + 6

def initialize_fireflies(num_fireflies, num_dimensions, domain=(-10, 10)):
    return (domain[1] - domain[0]) * np.random.rand(num_fireflies, num_dimensions) + domain[0]

def move_firefly(firefly, attractive_firefly, alpha, step_size):
    return firefly + alpha * (attractive_firefly - firefly) + step_size * np.random.rand(firefly.shape[0])

def firefly_algorithm(num_fireflies, num_dimensions, max_iterations, alpha0, alpha_dampening, beta0, gamma, step_size):
    domain = (-10, 10)
    fireflies = initialize_fireflies(num_fireflies, num_dimensions, domain)

    for iteration in range(max_iterations):
        for i in range(num_fireflies):
            for j in range(num_fireflies):
                if objective_function(fireflies[i]) < objective_function(fireflies[j]):
                    attractive_firefly = fireflies[j]
                    fireflies[i] = move_firefly(fireflies[i], attractive_firefly, update_alpha(alpha0, alpha_dampening, iteration, max_iterations), step_size)

    best_solution = fireflies[np.argmin([objective_function(firefly) for firefly in fireflies])]

    return best_solution

def update_alpha(alpha0, alpha_dampening, iteration, max_iterations):
    return alpha0 * (1.0 - iteration / max_iterations) ** alpha_dampening

if __name__ == "__main__":
    num_fireflies = 20
    num_dimensions = 1
    max_iterations = 100
    alpha0 = 0.2
    alpha_dampening = 2.0
    beta0 = 1.0
    gamma = 1.0
    step_size = 0.01

    optimal_solution = firefly_algorithm(num_fireflies, num_dimensions, max_iterations, alpha0, alpha_dampening, beta0, gamma, step_size)

    print("Optimal Solution:", optimal_solution)
    print("Optimal Value:", objective_function(optimal_solution))

    # Visualize optimization process
    x_values = np.linspace(-10, 10, 100)
    y_values = objective_function(x_values)
    plt.plot(x_values, y_values, label='Objective Function')
    plt.scatter(optimal_solution, objective_function(optimal_solution), color='red', label='Optimal Solution')
    plt.title('Firefly Algorithm Optimization')
    plt.xlabel('X')
    plt.ylabel('Objective Function Value')
    plt.legend()
    plt.show()

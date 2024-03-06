# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 14:48:05 2024

@author: francois
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d

def optimize_fiting_2_curves(y, t, y_goal, t_goal, initial_params=[1, 1, 0]):
    """
    Optimize the fitting of the curve y to the goal curve y_goal by minimizing a cost function.
    
    Arguments:
    y : array of y values of the original curve
    t : array of t values of the original curve
    y_goal : array of y values of the goal curve
    t_goal : array of t values of the goal curve
    initial_params : list of initial parameter values [amplitude, widening, offset]
    
    amplitude_optimal, widening_optimal, offset_optimal : optimal parameter values after optimization
    
    Returns:
    y_prime_optimal, t_prime_optimal : optimal values of the original curve after adjustment
    """
    
    # Function to adjust the curve
    def function(y, t, amplitude=1, widening=1, offset=0):
        y_prime = y * amplitude # adjust the y axis 
        t_prime = t * widening + offset # adjust the x axis
        return y_prime, t_prime

    # Cost function to minimize
    def cost_function(params):
        # counter to keep track of the dysplayed loss
        cost_function.i += 1
        
        amplitude, widening, offset = params
        y_prime, t_prime = function(y, t, amplitude, widening, offset)

        # Interpolate y to have the same number of elements as y_goal
        interpolator = interp1d(t_prime, y_prime, kind='quadratic', bounds_error=False, fill_value="extrapolate")
        y_interpolated = interpolator(t_goal)
        # same thing as above, having the same number of elements as t_goal
        t_interpolated = np.linspace(t_prime[0], t_prime[-1], num=len(t_goal),endpoint=True)

        # Calculate the differences between the two data sets (in amplitude and frames)
        difference = np.sum((y_interpolated - y_goal)**2) + np.sum((t_interpolated-t_goal)**2)
        
        # Print the loss every ten iterations
        if cost_function.i % 10 == 0:
            print("Iteration:", cost_function.i, "Loss:", difference)
        return difference
    
    # counter init
    cost_function.i = 0
    # Minimization of the cost function 
    # those numbers fit my usage, you might want to change it 
    # amplitude > 0.5 ; widening > 0.5 ; -1000 < offset < 1000
    bounds = ((0.5, None), (0.5, None), (-1000,1000)) 
    result = minimize(cost_function, initial_params,bounds=bounds, method='L-BFGS-B')


    # Optimal parameter values
    amplitude_optimal, widening_optimal, offset_optimal = result.x
    print(result.message)
    # Calculate the optimal values of the original curve after adjustment
    y_prime_optimal, t_prime_optimal = function(y, t, amplitude=amplitude_optimal, widening=widening_optimal, offset=offset_optimal)
    
    # Display results
    print("Optimal parameter values:")
    print("Optimal Amplitude:", amplitude_optimal)
    print("Optimal Widening:", widening_optimal)
    print("Optimal Offset:", offset_optimal)
    
    return y_prime_optimal, t_prime_optimal


# Example usage:
#------------------------------------------------------------------------------
if __name__ == "__main__":
    
    # Data of the curves
    y_original = np.array([0, 2, 3, 1])
    t_original = np.array([0, 1, 2, 3])
    y_goal = np.array([0, 1, 1.7, 3, 4.2, 4.8, 3.3, 2.7, 1.5, 1, 0.475, 0])*8
    t_goal = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])+6
    
    # Calling the optimize_fiting_2_curves function
    y_prime_optimal, t_prime_optimal = optimize_fiting_2_curves(y_original, t_original, y_goal, t_goal)
    
    
    # Display curves
    plt.plot(t_original, y_original, "go", label="Original")
    plt.plot(t_prime_optimal, y_prime_optimal, "bo", label="Adjusted")
    plt.plot(t_goal, y_goal, "yo", label="Goal")
    plt.title("Example data and possible usage")
    plt.ylabel("Amplitude")
    plt.xlabel("Frames")
    plt.legend()
    plt.grid()
    plt.show()
    

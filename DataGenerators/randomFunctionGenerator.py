# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 23:13:04 2024

@author: Abram Nothnagle and ChatGPT4o
"""

import random
import pandas as pd

# Logical operators dictionary
logical_operators = ['and','and not','or','or not','^']

# Generate a random logical function
def generate_random_function(n):
    conditions = []
    for i in range(1,n+1):  # Random number of conditions
        var1 = f"x{i}"
        if random.random() < 0.5:
            # Compare variable to a threshold
            threshold = round(random.uniform(0.1, 0.5), 2)
            condition = f"{var1} < {threshold}"
        else:
            # Compare variable to another variable
            var2 = f"x{random.randint(1, n)}"
            condition = f"{var1} < {var2}"
        conditions.append(condition)

    # Combine conditions with random logical operators
    function = conditions[0]
    for condition in conditions[1:]:
        #op = random.choice(list(logical_operators.keys()))
        op = random.choice(logical_operators)
        function = f"({function}) {op} ({condition})"

    return function

# Evaluate the logical function
def evaluate_function(func, inputs):
    local_vars = {f"x{i+1}": inputs[i] for i in range(len(inputs))}
    try:
        return eval(func, local_vars)
    except Exception as e:
        print(f"Error evaluating function: {e}")
        return 0

# Generate dataset
def generate_dataset(n, samples, func):
    data = []
    for _ in range(samples):
        inputs = [random.uniform(0, 1) for _ in range(n)]
        output = int(evaluate_function(func, inputs))
        data.append(inputs + [output])
    return data

# Main script
def main():
    try:
        n = int(input("Enter the number of variables (n): "))
        samples = int(input("Enter the number of samples: "))
        lopsided = input("Allow lopsided data? [y/n]: ")
        
        # If the users doesn't want lopsided data, need to make sure 
        # that somewhere between 40% and 60% of the data evaluate as True
        if lopsided == 'n':
            hits = 0
            while hits < 0.4*samples or hits > 0.6*samples:
                # Generate a random logical function
                random_function = generate_random_function(n)
                print(f"Generated Logical Function: {random_function}")
                
                # Create dataset
                dataset = generate_dataset(n, samples, random_function)
                #print(dataset)
        
                # Check the number of True evaluations to determine if lopsided data
                columns = [f"x{i+1}" for i in range(n)] + ["y"]
                df = pd.DataFrame(dataset, columns=columns)
                hits = df['y'].sum()
                print(hits)
        else:
            # Generate a random logical function
            random_function = generate_random_function(n)
            print(f"Generated Logical Function: {random_function}")
            
            # Create dataset
            dataset = generate_dataset(n, samples, random_function)
            #print(dataset)
            
            # Store data in dataframe
            columns = [f"x{i+1}" for i in range(n)] + ["y"]
            df = pd.DataFrame(dataset, columns=columns)
    
        # Save to CSV
        output_file = f"logical_fn_dataset_{n}_{samples}_lopsided{lopsided}.csv"
        df.to_csv(output_file, index=False)
        print(f"Dataset saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

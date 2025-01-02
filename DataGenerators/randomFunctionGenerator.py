# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 23:13:04 2024

@author: Abram Nothnagle and ChatGPT4o
"""

import random
import pandas as pd
import hashlib
import traceback

# Logical operators dictionary
logical_operators = ['and','and not','or','or not','^']
function_encode = ''

# Generate a random logical function
def generate_random_function(n):
    conditions = []
    encodes = []
    for i in range(1,n+1):  # Random number of conditions
        var1 = f"x{i}"
        if random.random() < 0.5:
            # Compare variable to a threshold
            threshold = round(random.uniform(0.1, 0.5), 2)
            if random.uniform(0,1) <= 0.5:
                compare = "<"
                enc_compare = "L"
            else:
                compare = ">"
                enc_compare = "G"
            condition = f"{var1} {compare} {threshold}"
            str_threshold = str(threshold).replace('.', '')
            enc = f"{var1}{enc_compare}{str_threshold}"
        else:
            # Compare variable to another variable
            i2 = random.randint(1,n)
            var2 = f"x{i2}"
            condition = f"{var1} < {var2}"
            enc = f"{var1}L{var2}"
        conditions.append(condition)
        encodes.append(enc)

    # Combine conditions with random logical operators
    function = conditions[0]
    encoded_function= encodes[0]
    for i in range(1,len(conditions)):
        condition = conditions[i]
        encoded = encodes[i]
        #op = random.choice(list(logical_operators.keys()))
        op = random.choice(logical_operators)
        op_idx = logical_operators.index(op) #I know there's a better way to have done this, but performance doesn't matter
        function = f"({function}) {op} ({condition})"
        encoded_function = f"{encoded_function}{op_idx}{encoded}-"

    return function, encoded_function

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

# Encode function using sha256
def encode_function(logical_function):
    # Use SHA256 for hashing
    hash_object = hashlib.sha256(logical_function.encode())
    # Convert hash to a shorter representation (e.g., first 8 characters of the hex digest)
    encoded = hash_object.hexdigest()[:8]
    return encoded

# Deterministic script
def deterministicGen():
    print('Generating deterministic data...')
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
                random_function, encoded_fn = generate_random_function(n)
                print(f"Generated Logical Function: {random_function}")
                
                # Create dataset
                dataset = generate_dataset(n, samples, random_function)
                #print(dataset)
        
                # Check the number of True evaluations to determine if lopsided data
                columns = [f"x{i+1}" for i in range(n)] + ["y1"]
                df = pd.DataFrame(dataset, columns=columns)
                hits = df['y1'].sum()
                print(hits)
        else:
            # Generate a random logical function
            random_function, encoded_fn = generate_random_function(n)
            print(f"Generated Logical Function: {random_function}")
            
            # Create dataset
            dataset = generate_dataset(n, samples, random_function)
            #print(dataset)
            
            # Store data in dataframe
            columns = [f"x{i+1}" for i in range(n)] + ["y1"]
            df = pd.DataFrame(dataset, columns=columns)
        
        encoded_hash = encode_function(random_function)
        # Save to CSV
        # Save to CSV with hash
        output_file = f"logi_fn_dataset_{n}_{samples}_lop{lopsided}_{encoded_hash}.csv"
        # Save to CSV with encoded name
        #output_file = f"logi_fn_dataset_{n}_{samples}_lop{lopsided}_{encoded_fn}.csv"
        df.to_csv(output_file, index=False)
        print(f"Dataset saved to {output_file}")
        
        # Save the hash to the hash file
        hashes = pd.read_csv('randomFunctionGenerator_Hashes.csv')
        new_hash = pd.DataFrame({'Hash':[encoded_hash], 'Function':[random_function]})
        hashes = pd.concat([hashes, new_hash], ignore_index = True)
        hashes.to_csv('randomFunctionGenerator_Hashes.csv', index=False)
        print(encoded_hash)
    except Exception as e:
        print(f"An error occurred: {e}")

# Noisy script
# Same as deterministic, except it adds some unform noise to the data
# that gets stored. The output will be calculated deterministically, rather
# the data that is returned has noise applied after the logic value is calculated
def noisyGen():
    print('Generating noisy data...')
    try:
        n = int(input("Enter the number of variables (n): "))
        samples = int(input("Enter the number of samples: "))
        noise = float(input("Enter the noise to add to the data (<1, normal values ~0.001): "))
        lopsided = input("Allow lopsided data? [y/n]: ")
        
        # If the users doesn't want lopsided data, need to make sure 
        # that somewhere between 40% and 60% of the data evaluate as True
        if lopsided == 'n':
            hits = 0
            while hits < 0.4*samples or hits > 0.6*samples:
                # Generate a random logical function
                random_function, encoded_fn = generate_random_function(n)
                print(f"Generated Logical Function: {random_function}")
                
                # Create dataset
                dataset = generate_dataset(n, samples, random_function)
                
                #If noise was applied more than 0, adds uniform noise from -noise to noise
                if noise > 0.0:
                    noisy_dataset = [[element + random.uniform(-noise, noise) for element in row] for row in dataset]

                # Check the number of True evaluations to determine if lopsided data
                columns = [f"real_x{i+1}" for i in range(n)] + ["y1"]
                real_df = pd.DataFrame(dataset, columns=columns)
                noisy_columns = [f"x{i+1}" for i in range(n)] + ["garbage"]
                noisy_data = pd.DataFrame(noisy_dataset, columns=noisy_columns)
                df = pd.concat([real_df, noisy_data], axis = 1)
                # Reorder the columns
                combined_column_order = [f"real_x{i+1}" for i in range(n)] + [f"x{i+1}" for i in range(n)] + ["y1"]
                df = df[combined_column_order]
                hits = df['y1'].sum()
                print(hits)
        else:
            # Generate a random logical function
            random_function, encoded_fn = generate_random_function(n)
            print(f"Generated Logical Function: {random_function}")
            
            # Create dataset
            dataset = generate_dataset(n, samples, random_function)
            
            #If noise was applied more than 0, adds uniform noise from -noise to noise
            if noise > 0.0:
                noisy_dataset = [[element + random.uniform(-noise, noise) for element in row] for row in dataset]

            # Store data in dataframe
            columns = [f"real_x{i+1}" for i in range(n)] + ["y1"]
            real_df = pd.DataFrame(dataset, columns=columns)
            noisy_columns = [f"x{i+1}" for i in range(n)] + ["garbage"]
            noisy_data = pd.DataFrame(noisy_dataset, columns=noisy_columns)
            df = pd.concat([real_df, noisy_data], axis = 1)
            # Reorder the columns
            combined_column_order = [f"real_x{i+1}" for i in range(n)] + [f"x{i+1}" for i in range(n)] + ["y1"]
            df = df[combined_column_order]
        
        encoded_hash = encode_function(random_function)
        # Save to CSV
        # Save to CSV with hash
        output_file = f"logi_fn_dataset_{n}_{samples}_lop{lopsided}_{encoded_hash}.csv"
        # Save to CSV with encoded name
        #output_file = f"logi_fn_dataset_{n}_{samples}_lop{lopsided}_{encoded_fn}.csv"
        df.to_csv(output_file, index=False)
        print(f"Dataset saved to {output_file}")
        
        # Save the hash to the hash file
        hashes = pd.read_csv('randomFunctionGenerator_Hashes.csv')
        new_hash = pd.DataFrame({'Hash':[encoded_hash], 'Function':[random_function]})
        hashes = pd.concat([hashes, new_hash], ignore_index = True)
        hashes.to_csv('randomFunctionGenerator_Hashes.csv', index=False)
        print(encoded_hash)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

def main():
    noisyGen()
    #deterministicGen()

if __name__ == "__main__":
    main()

import pandas as pd
import itertools
import numpy as np

### In the following docstrings, the values are given just to exemplify the format of the output, not to give the actual "real" output

steps = [1, 2, 3, 4]

def get_missing_steps(row, steps):
    """
    Get the steps that are PRESENT (not missing) as a tuple for each row.
    Note: The CSV columns are named 'stepX_missing'. 
    If step1_missing is 0, it means step 1 is PRESENT.
    If step1_missing is 1, it means step 1 is MISSING.
    
    Example:
        Input: row = {'step1_missing': 1, 'step2_missing': 0, 'step3_missing': 1, 'step4_missing': 0},
               steps = [1, 2, 3, 4]
        Output: (2, 4)  <-- These are the steps that were used (missing=0)
    """
    present_steps = [] 
    ##############################################################################
    ### STUB: INSERT CODE HERE: Get missing steps as a tuple for each row.###
    ##############################################################################
    
    # Iterate through the steps 1, 2, 3, 4
    for step in steps:
        # Check the column 'stepX_missing'. 
        # If the value is 0, the step was INCLUDED (Present) in the prompt.
        # If the value is 1, the step was OMITTED (Missing).
        col_name = f"step{step}_missing"
        if row[col_name] == 0:
            present_steps.append(step)

    # Return as a sorted tuple so it can be used as a dictionary key later
    return tuple(sorted(present_steps))

def generate_all_subsets(steps):
    """
    Generate all possible subsets for the steps.

    Example:
        Input: steps = [1, 2, 3]
        Output: [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    """
    all_subsets_missing = []
    for r in range(len(steps) + 1):
        subsets_r = list(itertools.combinations(steps, r))
        all_subsets_missing.extend(subsets_r)
    return all_subsets_missing

def compute_v_S(df, all_subsets_missing):
    """
    Compute v(S) for all subsets of steps.
    v(S) is the average accuracy (mean of is_correct) for a specific subset S.

    Example:
        Input: 
            df = pd.DataFrame({'missing_steps': [(), (1,), (2,), (1, 2)], 'is_correct': [0.8, 0.7, 0.6, 0.5]})
            all_subsets_missing = [(), (1,), (2,), (1, 2)]
        Output: 
            v_S = {(): 0.8, (1,): 0.7, (2,): 0.6, (1, 2): 0.5}
    """
    ##################################################################################
    ### STUB: INSERT CODE HERE: Compute v(S) for all subsets of missing steps.###
    ##################################################################################
    
    v_S = {}
    # Group the dataframe by the set of steps used (which we stored in 'missing_steps' column)
    # and calculate the mean of the 'is_correct' column for each group.
    grouped = df.groupby('missing_steps')['is_correct'].mean()
    
    # Convert to dictionary
    v_S = grouped.to_dict()
    
    return v_S

def compute_marginal_contributions(steps, v_S):
    """
    Compute the marginal contributions for each step.

    Example:
        Input:
            steps = [1, 2]
            v_S = {(): 0.85, (1,): 0.67, (2,): 0.72, (1, 2): 0.75}
        Output:
            Delta_sum = {1: 0.08, 2: 0.12}, valid_permutations_count = 2
    """
    permutations = list(itertools.permutations(steps))
    Delta_sum = {i: 0.0 for i in steps}
    valid_permutations_count = 0

    total_steps_set = set(steps)

    for pi in permutations:
        valid_permutation = True
        for i in steps:
            # Find the index of step i in the current permutation pi
            idx_i = pi.index(i)
            
            #############################################################################################
            ### STUB: INSERT CODE HERE: Retrieve S_i, S_i_union_i, S_i_sorted, S_i_union_i_sorted###
            #############################################################################################
            
            # S_i is the set of players (steps) that precede i in the ordering pi
            S_i = pi[:idx_i]
            
            # S_i_union_i is S_i plus the current step i
            S_i_union_i = pi[:idx_i+1]
            
            # Convert to sorted tuples to look them up in our v_S dictionary
            missing_S_i_sorted = tuple(sorted(S_i))
            missing_S_i_union_i_sorted = tuple(sorted(S_i_union_i))

            v_S_i = v_S.get(missing_S_i_sorted, np.nan)
            v_S_i_union_i = v_S.get(missing_S_i_union_i_sorted, np.nan)
            
            if np.isnan(v_S_i) or np.isnan(v_S_i_union_i):
                valid_permutation = False
                break
            else:
                ###############################################################################
                ### STUB: INSERT CODE HERE: Compute the marginal contribution of step i###
                ###############################################################################
                
                # Marginal contribution = Value with step i - Value without step i
                marginal_contribution = v_S_i_union_i - v_S_i
                Delta_sum[i] += marginal_contribution
                
        if valid_permutation:
            valid_permutations_count += 1
    return Delta_sum, valid_permutations_count

def compute_shapley_values(Delta_sum, valid_permutations_count, steps):
    """
    Compute the Shapley values for each step.

    Example:
        Input: 
            Delta_sum = {1: 0.08, 2: 0.12}
            valid_permutations_count = 2
            steps = [1, 2]
        Output: 
            Shapley_values = {1: 0.04, 2: 0.06}
    """
    ##############################################################
    ### STUB: INSERT CODE HERE: Compute the Shapley values###
    ##############################################################
    
    shapley_values = {}
    for step in steps:
        # The Shapley value is the average of the marginal contributions over all permutations
        if valid_permutations_count > 0:
            shapley_values[step] = Delta_sum[step] / valid_permutations_count
        else:
            shapley_values[step] = 0.0
            
    return shapley_values

def main():
    # Load the data
    try:
        df = pd.read_csv('evaluation_with_steps.csv')
    except FileNotFoundError:
        print("Error: 'evaluation_with_steps.csv' not found. Please make sure it is in the same directory.")
        return

    # Apply the function to identify which steps are present in each row
    df['missing_steps'] = df.apply(get_missing_steps, axis=1, args=(steps,))

    ######################################################################################
    # STUB: INSERT CODE HERE: Generate all possible subsets for the missing steps #       (1 line of code)
    ######################################################################################
    all_subsets_missing = generate_all_subsets(steps)

    ###############################################
    # STUB: INSERT CODE HERE: Compute v(S) #    (1 line of code)
    ###############################################
    v_S = compute_v_S(df, all_subsets_missing)

    #############################################################
    # STUB: INSERT CODE HERE: Compute the Shapley values and print for each step #
    #############################################################
    
    # 1. Compute marginal contributions (sum of deltas)
    Delta_sum, valid_permutations_count = compute_marginal_contributions(steps, v_S)
    
    # 2. Compute final Shapley values (average of deltas)
    shapley_values = compute_shapley_values(Delta_sum, valid_permutations_count, steps)
    
    # 3. Print the results
    print("Shapley Values for each step:")
    for step, val in shapley_values.items():
        print(f"Step {step}: {val:.4f}")

if __name__ == "__main__":
    main()

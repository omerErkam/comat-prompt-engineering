import pandas as pd
import itertools
import numpy as np

### In the following docstrings, the values are given just to exemplify the format of the output, not to give the actual "real" output

steps = [1, 2, 3, 4]

def get_missing_steps(row, steps):
    """
    Get missing steps as a tuple for each row.
    
    Example:
        Input: row = {'step1_missing': 1, 'step2_missing': 0, 'step3_missing': 1, 'step4_missing': 0},
               steps = [1, 2, 3, 4]
        Output: (2, 4)
    """
    missing_steps = []
    ##############################################################################
    ### STUB: INSERT CODE HERE: Get missing steps as a tuple for each row.###
    ##############################################################################

    raise NotImplementedError("Get missing steps as a tuple for each row.")
    return tuple(sorted(missing_steps))

def generate_all_subsets(steps):
    """
    Generate all possible subsets for the missing steps.

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
    Compute v(S) for all subsets of missing steps.

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

    raise NotImplementedError("Implement the code to compute v(S) for all subsets of missing steps.")
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
            idx_i = pi.index(i)
            #############################################################################################
            ### STUB: INSERT CODE HERE: Retrieve S_i, S_i_union_i, S_i_sorted, S_i_union_i_sorted###
            #############################################################################################

            raise NotImplementedError("Implement the retrieval of S_i, S_i_union_i, and their sorted tuples.")
            v_S_i = v_S.get(missing_S_i_sorted, np.nan)
            v_S_i_union_i = v_S.get(missing_S_i_union_i_sorted, np.nan)
            if np.isnan(v_S_i) or np.isnan(v_S_i_union_i):
                valid_permutation = False
                break
            else:
                ###############################################################################
                ### STUB: INSERT CODE HERE: Compute the marginal contribution of step i###
                ###############################################################################

                raise NotImplementedError("Implement the computation of the marginal contribution of step i.")
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

    raise NotImplementedError("INSERT CODE HERE: Compute the Shapley values")
    return shapley_values

def main():
    df = pd.read_csv('evaluation_with_steps.csv')

    df['missing_steps'] = df.apply(get_missing_steps, axis=1, args=(steps,))

    ######################################################################################
    # STUB: INSERT CODE HERE: Generate all possible subsets for the missing steps #       (1 line of code)
    ######################################################################################

    raise NotImplementedError("Implement the code to generate all possible subsets for the missing steps.")

    ###############################################
    # STUB: INSERT CODE HERE: Compute v(S) #    (1 line of code)
    ###############################################

    raise NotImplementedError("Implement the code to compute v(S).")

    #############################################################
    # STUB: INSERT CODE HERE: Compute the Shapley values and print for each step #
    #############################################################

    raise NotImplementedError("Implement the code to compute the Shapley values.")

    

if __name__ == "__main__":
    main()

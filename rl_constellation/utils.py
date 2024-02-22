def get_local_and_neighboring_benefits(benefits, i, M, L_hat):
    """
    Given a benefit matrix, gets the local benefits for agent i.
        Filters the benefit matrices to only include the top M tasks for agent i.

    Also returns the neighboring benefits for agent i.
        A (n-1) x M x L_hat matrix with the benefits for every other agent for the top M tasks.

    Finally, returns the global benefits for agent i.
        A (n-1) x (m-M) x L_hat matrix with the benefits for all the other (m-M) tasks for all other agents.
    """
    pass
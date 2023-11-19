# federated-deep-survival-analysis

Improvements

1. Add possibility to select activation function
2. Add possibility to add Drop-out and Batch-norm
3. Control coherency in auton-survival pre-processors
4. Test on different SUPPORT datasets
5. Test on other non-recurrent datasets 

Next steps

1. Federate deepcoxph with Concordance Index Censored as additional metric
2. Aggregate ibs / ipcw client-side where possible
3. Explore federated epochs (5+ until convergence = test metric worsens or constant) and client epochs ([1, 10, 20, 30]) + client size distribution (stratified also on number of censoring, hub=[70, 10, 10, 10], small_guy=[30, 30, 30, 10], fair=[25, 25, 25, 25]) all hyperparameters with 10 random seeds in the first split centralized (10-20%) + 4

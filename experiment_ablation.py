from ExplanationEvaluation.configs.selector import Selector
from ExplanationEvaluation.tasks.replication import replication

_dataset = 'treecycles' # One of: bashapes, bacommunity, treecycles, treegrids, ba2motifs, mutag
_explainer = 'pgexplainer' # One of: pgexplainer, gnnexplainer


# PGExplainer
config_path = f"./ExplanationEvaluation/configs/replication/explainers/{_explainer}/{_dataset}.json"

config = Selector(config_path).args.explainer

# Permutations
coef_size = [10, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.0001]
coef_entr = [10, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.0001]

### QUICKER VERSION
# coef_size = [1.0, 0.01, 0.0001]
# coef_entr = [1.0, 0.01, 0.0001]
# config.seeds = [0]

results = []

for size in coef_size:
    for entropy in coef_entr:
        print(size, entropy)
        interim_resuts = {}

        config.reg_size = size
        config.reg_ent = entropy

        (auc, std), _ = replication(config, run_qual=False, results_store=False)

        interim_resuts["AUC"] = auc
        interim_resuts["std"] = std

        res = {
            'size': size,
            'entropy': entropy,
            'auc': auc,
            'std': std
        }
        results.append(res)
        
for r in results:
    print(r)
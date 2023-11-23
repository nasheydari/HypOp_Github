from src.run_exp import exp_centralized, exp_centralized_for
import json




with open('configs/Hypermaxcut_syn_new.json') as f:
   params = json.load(f)
exp_centralized(params)


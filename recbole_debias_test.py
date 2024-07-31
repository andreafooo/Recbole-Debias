from recbole.quick_start import run_recbole
from recbole_debias.quick_start import run_recbole_debias
import json
from time import time


# See evaluation metrics: https://recbole.io/docs/recbole/recbole.evaluator.metrics.html

def run_configurations(model):

    config_dict=["config_test.yaml"]



    output_dict = run_recbole_debias(model=model, 
                              config_file_list=config_dict)
   
 

run_configurations("PDA")


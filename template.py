import os
import logging
from pathlib import Path

#Name of the project
project_name = 'store_demand_forecasting'

#list of files to be created for project
list_of_files = [
    
                f'src/__init__.py',
                f'src/components/data_ingestion.py',
                f'src/components/data_transformation.py',
                f'src/components/data_validation.py',
                f'src/components/data_evaluation.py',
                f'src/components/model_trainer.py',
                f'src/utils/__init__.py',
                f'src/utils/common.py',
                f'src/logger/__init__.py',
                f'src/exception/__init__.py',
                f'src/pipeline/__init__.py',
                f'src/pipeline/training_pipeline.py',
                f'src/pipeline/prediction_pipeline.py',
                f'src/constants/__init__.py',
                f'src/entity/__init__.py',
                f'src/entity/config_entity.py',
                f'end_point.py',
                f'app.py',
                f'main.py',
                f'Dockerfile', 
                f'requirements.txt', 
                f'setup.py', 
                f'research/leed_predictor_one.py', 
                
                ]



for filepath in list_of_files:

    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != '':
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory {filepath} for the file {filename}")


    if (not os.path.exists(filepath)) or  (os.path.getsize(filepath) == 0):
        with open(filepath,'w') as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    
    else:
        logging.info(f"{filename} is already exists")
        
import sys
from src.exception import NerException
from src.pipeline.training_pipeline import TrainPipeline
from src.constants import *


def training():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

    except Exception as e:
        raise NerException(e, sys) from e


if __name__ == "__main__":
    training()
from enum import Enum


class ClassifierEnum(str, Enum):
    xgboost = "xgboost"
    random_forest = "random_forest"

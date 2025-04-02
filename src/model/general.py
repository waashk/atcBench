
from src.model.llm import LLMClassifier
from src.model.slm import SLMClassifier
from src.model.traditional import TraditionalClassifier

def getClassifier(model_config, dataset):
    _type = model_config.type

    if _type == 'llm':
        clf = LLMClassifier(model_config, dataset)
    elif _type == 'slm':
        clf = SLMClassifier(model_config, dataset)
    elif _type == 'traditional':
        clf = TraditionalClassifier(model_config, dataset)

    return clf
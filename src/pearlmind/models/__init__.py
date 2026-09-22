from .ensemble import XGBoostModel


def load_model(path):
    model = XGBoostModel()
    model.load(path)
    return model

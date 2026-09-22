from pearlmind.evaluation import FairnessAuditor


class BaseModel:
    def __init__(self, name, version, enable_fairness_audit=True):
        self.name, self.version = name, version
        self.enable_fairness_audit = enable_fairness_audit
        self.params = {}
        self.is_fitted = False

    def audit_fairness(self, X, y, sensitive_features=None):
        return FairnessAuditor().audit(y, self.predict(X), sensitive_features)

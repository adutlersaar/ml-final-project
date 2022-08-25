import inspect
from sklearn.ensemble import RandomForestClassifier
from .utils import fit_with_time, fit_for_leave_out


class WrappedRandomForestClassifier(RandomForestClassifier):
    def __init__(self, *args, **kwargs):
        RandomForestClassifier.__init__(self, *args, **kwargs)
        self.org_fit = self.fit
        self.fit = self.fit_modified
        self.fit_time = None
        self.metrics = {}
        self.clf_name_ = 'RandomForestClassifier'

    def fit_modified(self, X, y, cv=False, **kwargs):
        return fit_for_leave_out(self, X, y, cv=cv, **kwargs) if cv else fit_with_time(self, X, y, **kwargs)


WrappedRandomForestClassifier.__init__.__signature__ = inspect.signature(RandomForestClassifier.__init__)

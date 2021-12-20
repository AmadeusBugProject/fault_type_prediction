import joblib
from sklearn.base import BaseEstimator, TransformerMixin

from artifact_detection_model.transformer.DoNotReplaceArtifacts import DoNotReplaceArtifacts
from artifact_detection_model.transformer.ReplaceButKeepExceptionNames import ReplaceButKeepExceptionNames
from artifact_detection_model.transformer.SimpleReplace import SimpleReplace
from file_anchor import root_dir

try:
    classifier = joblib.load(root_dir() + 'artifact_detection_model/out/artifact_detection.joblib')
except FileNotFoundError:
    raise RuntimeError("""
    ---------------
    No pretrained artifact detection model found!
    please check out https://github.com/AmadeusBugProject/artifact_detection
    and run artifact_detection_model/RUN_train_model.py and copy 
    artifact_detection_model/out/artifact_detection.joblib to this projects artifact_detection_model/out/ folder.
    ---------------""")

SIMPLE = 'simple'
KEEP_EXCEPTION_NAMES = 'keep_exception_names'
DO_NOT_REPLACE = 'no_replacements'

replacement_strategies = {SIMPLE: SimpleReplace(),
                          KEEP_EXCEPTION_NAMES: ReplaceButKeepExceptionNames(),
                          DO_NOT_REPLACE: DoNotReplaceArtifacts()}


class ArtifactRemoverTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, replacement_strategy=SIMPLE):
        self.replacement_strategy = replacement_strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.replacement_strategy == DO_NOT_REPLACE:
            return X
        return [self.predict_and_remove(i) for i in X]

    def predict_and_remove(self, issue):
        replacement_strategy = replacement_strategies[self.replacement_strategy]

        prediction = classifier.predict(issue.splitlines())
        text_indices = [i for i, e in enumerate(prediction) if e == 1]
        cleaned_issue = []
        for i in range(0, len(issue.splitlines())):
            if i in text_indices:
                cleaned_issue.append(issue.splitlines()[i])
            else:
                replacement = replacement_strategy.get_replacement(issue.splitlines()[i])
                if replacement.strip():
                    cleaned_issue.append(replacement)
        return '\n'.join(cleaned_issue)

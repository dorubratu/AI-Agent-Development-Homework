"""
L8 Task 3 - Intent Classifier (scikit-learn)
=============================================

Un text classifier TF-IDF + LogisticRegression care prezice intenția unui query
(search / extract / summarize). Pipeline clasic sklearn:

    TfidfVectorizer → LogisticRegression

Față de un LLM, e:
  • mult mai rapid (microsecunde vs secunde)
  • gratuit (rulează local, fără apel API)
  • dar limitat la clasele văzute la antrenare

Rulează (antrenează + salvează modelul):
    PYTHONPATH="skillab-py/src:src" python3 src/intent_classifier.py
"""
import logging
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from intent_data import TRAIN_DATA, TEST_DATA

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent / "models" / "intent_clf.joblib"


def build_pipeline() -> Pipeline:
    """TF-IDF (uni+bigrame) → LogisticRegression."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),      # unigrame + bigrame prind expresii ("dă-mi un rezumat")
            min_df=1,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=10.0,                  # set mic → regularizare mai relaxată
            class_weight="balanced",
        )),
    ])


def train(save: bool = True) -> Pipeline:
    """Antrenează clasificatorul pe TRAIN_DATA și (opțional) îl salvează."""
    X = [q for q, _ in TRAIN_DATA]
    y = [lbl for _, lbl in TRAIN_DATA]

    pipe = build_pipeline()
    pipe.fit(X, y)

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, MODEL_PATH)
        logger.info(f"Model salvat la {MODEL_PATH}")

    return pipe


def load() -> Pipeline:
    """Încarcă modelul salvat (antrenează dacă nu există)."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    logger.info("Modelul nu există încă — antrenez acum.")
    return train(save=True)


def predict(pipe: Pipeline, query: str) -> str:
    return pipe.predict([query])[0]


def predict_proba(pipe: Pipeline, query: str) -> dict:
    proba = pipe.predict_proba([query])[0]
    return dict(zip(pipe.classes_, (round(float(p), 3) for p in proba)))


def evaluate(pipe: Pipeline) -> float:
    """Accuracy pe setul de test (queries nevăzute la antrenare)."""
    correct = 0
    for q, expected in TEST_DATA:
        if predict(pipe, q) == expected:
            correct += 1
    return correct / len(TEST_DATA)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pipe = train(save=True)
    acc = evaluate(pipe)
    print(f"\nAntrenat pe {len(TRAIN_DATA)} exemple.")
    print(f"Accuracy pe test ({len(TEST_DATA)} exemple): {acc:.1%}\n")
    print("Exemple de predicții:")
    for q in [
        "Caută facturile de la TechSoft",
        "Extrage suma totală din contract",
        "Rezumă raportul trimestrial",
        "Ce contact are DataPro?",
    ]:
        print(f"  {q!r:45} → {predict(pipe, q):10} {predict_proba(pipe, q)}")

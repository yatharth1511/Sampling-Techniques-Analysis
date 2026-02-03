from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sampling import (
    random_sampling,
    cluster_sampling,
    bootstrap_sampling,
    stratified_sampling,
    strategic_sampling
)
from models import get_models

def train_and_evaluate(X, y):
    samplers = {
        "Sampling1": lambda X, y: random_sampling(X, y, n_samples=int(0.7 * len(X))),
        "Sampling2": cluster_sampling,
        "Sampling3": bootstrap_sampling,
        "Sampling4": stratified_sampling,
        "Sampling5": strategic_sampling
    }

    models = get_models()
    results = {}

    for s_name, sampler in samplers.items():
        sampled_data = sampler(X, y)

        X_s = sampled_data.drop("Class", axis=1)
        y_s = sampled_data["Class"]

        X_train, X_test, y_train, y_test = train_test_split(
            X_s, y_s, test_size=0.3, random_state=42, stratify=y_s
        )

        results[s_name] = {}

        for m_name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            results[s_name][m_name] = round(acc * 100, 2)

    return results

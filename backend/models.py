from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def get_models():
    return {
        "M1": LogisticRegression(max_iter=1000),
        "M2": DecisionTreeClassifier(),
        "M3": RandomForestClassifier(n_estimators=100),
        "M4": KNeighborsClassifier(),
        "M5": SVC()
    }

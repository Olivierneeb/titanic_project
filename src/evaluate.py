
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from preprocess import load_data, preprocess_data

def evaluate_model(model, X_test, y_test):
    """Évalue les performances du modèle."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report

def main():
    # Chargement et préparation des données
    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / 'data' / 'titanic.csv'
    df = load_data(file_path)
    df = preprocess_data(df)

    # Séparation des caractéristiques et de la variable cible
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # Division des données
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Chargement du modèle
    model_path = project_root / 'models' / 'titanic_model.pkl'
    model = joblib.load(model_path)

    # Évaluation du modèle
    accuracy, report = evaluate_model(model, X_test, y_test)
    print(f"Accuracy : {accuracy}")
    print(f"Classification Report :\n{report}")

if __name__ == "__main__":
    main()

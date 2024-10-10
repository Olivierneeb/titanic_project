import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from preprocess import load_data, preprocess_data  # Assurez-vous que le script 'data.py' contient ces fonctions

def train_model(X, y):
    """Entraîne un modèle de classification sur les données fournies."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

def main():
    
    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / 'data' / 'titanic.csv'

    # Chargement et préparation des données
    df = load_data(file_path)
    df = preprocess_data(df)

    # Séparation des caractéristiques et de la variable cible
    X = df.drop(columns='Survived')
    y = df['Survived']

    # Division des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraînement du modèle
    model = train_model(X_train, y_train)

    # Sauvegarde du modèle
    model_path = project_root / 'models' / 'titanic_model.pkl'
    joblib.dump(model, model_path)
    print("Modèle entraîné et sauvegardé avec succès.")

if __name__ == "__main__":
    main()
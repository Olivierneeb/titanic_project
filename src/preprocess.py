import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Charge le dataset à partir d'un fichier CSV."""
    df = pd.read_csv(file_path)
    return df


def one_hot_encoding(df : pd.DataFrame, columns : list ):
    df = pd.get_dummies(df,
                        columns=columns,
                        drop_first=True)
    return df

def standardisation(df, columns: list):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def preprocess_data(df):
    """Prétraite les données """
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    df['Parch'] = df['Parch'].apply(lambda x : 1 if x > 0 else 0)
    df['SibSp'] = pd.cut(df['SibSp'],
                        bins=[-1, 0, 1, float('inf')], # intervalles (-1,0], (0,1] et (1,inf]
                        labels=['0', '1', '2+'])
    
    # One Hot encodage 
    df = one_hot_encoding(df, ['Pclass', 'Sex', 'Embarked','SibSp'])

    # Standardisation
    df = standardisation(df, ['Age', 'Fare'])

    return df


if __name__ == "__main__":
    df = load_data("../data/titanic.csv")
    # Supprimer les colonnes non pertinentes
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    print(df.head())
    print(f"nombre de lignes incomplètes : {df.isnull().any(axis=1).sum()}")
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(df.head())
    print(f"nombre de lignes incomplètes : {df.isnull().any(axis=1).sum()}")
    df = pd.get_dummies(df,
                        columns=['Pclass', 'Sex', 'Embarked'],
                        drop_first=True)
    df['Parch'] = df['Parch'].apply(lambda x : 1 if x > 0 else 0)
    df['SibSp'] = pd.cut(df['SibSp'],
                        bins=[-1, 0, 1, float('inf')], # intervalles (-1,0], (0,1] et (1,inf]
                        labels=['0', '1', '2+'])
    df = pd.get_dummies(df,
                        columns=['SibSp'],
                        drop_first=True)

    scaler = StandardScaler()
    df['Age'] = scaler.fit_transform(df[['Age']])
    df['Fare'] = scaler.fit_transform(df[['Fare']])

    print(df.head())

    print(preprocess_data(load_data("../data/titanic.csv")).head())

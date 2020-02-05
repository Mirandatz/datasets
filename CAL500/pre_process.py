from pathlib import Path
from typing import List

import arff as arff_output
import numpy as np
import pandas as pd
from scipy.io import arff as arff_input
from sklearn.model_selection import KFold as Folder

DATASET_NAME = "CAL500"
FEATURE_COUNT = 68
CLASS_COUNT = 174

FOLD_COUNT = 10


def get_folds_indices(data: pd.DataFrame, labels: pd.DataFrame) -> List[np.ndarray]:
    skf = Folder(n_splits=FOLD_COUNT, shuffle=True, random_state=42)
    return list(skf.split(X=data, y=labels))


def get_fold_dir(fold_nr: int) -> Path:
    return Path.cwd() / 'folds' / str(fold_nr)


def save_arff(df: pd.DataFrame, file_path: Path):
    attributes = [(f"Attr{i}", 'NUMERIC') for i in range(FEATURE_COUNT)]
    attributes += [(f"Class{i}", ['0', '1']) for i in range(CLASS_COUNT)]
    instance_count = df.shape[0]
    arff_data = [df.iloc[i].tolist() for i in range(instance_count)]
    arff_dict = {'attributes': attributes,
                 'data': arff_data,
                 'relation': DATASET_NAME,
                 'description': ''}

    with file_path.open(mode='wt') as file:
        arff_output.dump(obj=arff_dict, fp=file)


def save_train(data: pd.DataFrame, labels: pd.DataFrame, train_indices: np.ndarray, fold_dir: Path):
    train_data = data.iloc[train_indices]
    train_data.to_csv(fold_dir / 'train-data.csv', header=False, index=False)

    train_labels = labels.iloc[train_indices]
    train_labels.to_csv(fold_dir / 'train-labels.csv', header=False, index=False)

    train_data_and_labels = pd.concat([train_data, train_labels], axis='columns').reset_index(drop=True)
    save_arff(df=train_data_and_labels, file_path=fold_dir / 'train.arff')


def save_test(data: pd.DataFrame, labels: pd.DataFrame, test_indices: np.ndarray, fold_dir: Path):
    test_data = data.iloc[test_indices]
    test_data.to_csv(fold_dir / 'test-data.csv', header=False, index=False)

    test_labels = labels.iloc[test_indices]
    test_labels.to_csv(fold_dir / 'test-labels.csv', header=False, index=False)

    test_data_and_labels = pd.concat([test_data, test_labels], axis='columns').reset_index(drop=True)
    save_arff(df=test_data_and_labels, file_path=fold_dir / 'test.arff')


def main():
    # Reading the .arff and extracting and converting it to a pandas DataFarame
    arff_data = arff_input.loadarff('original-files/CAL500.arff')
    arff_df = pd.DataFrame(arff_data[0])
    df = pd.DataFrame(arff_df.values)
    df = df.apply(pd.to_numeric)

    # Splitting data and labels
    data = df.iloc[:, :FEATURE_COUNT].reset_index(drop=True)
    labels = df.iloc[:, FEATURE_COUNT:].reset_index(drop=True)
    labels = labels.astype('category')

    # Creating folds
    folds_indices = get_folds_indices(data, labels)

    # Saving
    for fold_nr, (train_indices, test_indices) in enumerate(folds_indices):
        fold_dir = get_fold_dir(fold_nr)
        print(f"Creating fold {fold_nr}...", end=' ')
        fold_dir.mkdir(parents=True, exist_ok=True)

        save_train(data=data, labels=labels, train_indices=train_indices, fold_dir=fold_dir)
        save_test(data=data, labels=labels, test_indices=test_indices, fold_dir=fold_dir)
        print("Done")

    print("Done...")


if __name__ == '__main__':
    main()

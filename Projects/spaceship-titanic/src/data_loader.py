
import pandas as pd
from pathlib import Path

def load_data(data_dir: str | Path = "data") -> dict[str, pd.DataFrame]:
    """
    Загружает train.csv, test.csv и sample_submission.csv.
    Возвращает словарь: {"train": df_train, "test": df_test, "submission": df_sub}
    """
    data_dir = Path(data_dir)
    
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sub_path = data_dir / "sample_submission.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    submission = pd.read_csv(sub_path)

    return {
        "train": train,
        "test": test,
        "submission": submission,
    }
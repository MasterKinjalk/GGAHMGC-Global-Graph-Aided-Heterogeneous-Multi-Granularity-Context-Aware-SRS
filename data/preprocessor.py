import os
import pickle
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from datetime import timedelta
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools

# --- New Imports for Parallelism ---
from pandarallel import pandarallel
from joblib import Parallel, delayed
import multiprocessing

# --- Initialize pandarallel ---
# This will allow us to use .parallel_apply() on DataFrames
pandarallel.initialize(progress_bar=True, nb_workers=multiprocessing.cpu_count())


def _process_graph_chunk(
    session_items: List[int], window_size: int
) -> Tuple[Dict, Dict]:
    """Helper function to process one session for graph creation. Runs in a separate process."""
    cooccurrence = defaultdict(lambda: defaultdict(int))
    transition = defaultdict(lambda: defaultdict(int))

    # Transitions (item_i -> item_{i+1})
    for i in range(len(session_items) - 1):
        transition[session_items[i]][session_items[i + 1]] += 1

    # Co-occurrence within a sliding window
    for i in range(len(session_items)):
        for j in range(i + 1, min(i + window_size + 1, len(session_items))):
            u, v = session_items[i], session_items[j]
            cooccurrence[u][v] += 1
            cooccurrence[v][u] += 1

    return dict(cooccurrence), dict(transition)


class BasePreprocessor(ABC):
    """
    Abstract Base Class for session data preprocessing.
    (Code from previous optimization is kept, with parallel versions of methods)
    """

    def __init__(self, config: Dict):
        self.config = config
        self.min_session_length: int = config["dataset"]["min_session_length"]
        self.min_item_frequency: int = config["dataset"]["min_item_frequency"]
        self.test_days: int = config["dataset"]["test_days"]
        self.validation_split: float = config["dataset"]["validation_split"]
        self.cooccurrence_window: int = config["dataset"].get("cooccurrence_window", 3)

    @abstractmethod
    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def create_sessions(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    def filter_items(self, sessions_df: pd.DataFrame) -> pd.DataFrame:
        """Filter out items with low frequency and sessions that become too short."""
        print("Filtering items by frequency...")

        all_items = itertools.chain.from_iterable(sessions_df["items"])
        item_counts = Counter(all_items)

        valid_items = {
            item
            for item, count in item_counts.items()
            if count >= self.min_item_frequency
        }

        print("Applying item filter to sessions (in parallel)...")
        # --- Use parallel_apply instead of apply ---
        sessions_df["items"] = sessions_df["items"].parallel_apply(
            lambda session_items: [
                item for item in session_items if item in valid_items
            ]
        )

        sessions_df = sessions_df[
            sessions_df["items"].str.len() >= self.min_session_length
        ].reset_index(drop=True)

        return sessions_df

    def create_item_mapping(
        self, sessions_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
        """Create integer mappings for item IDs."""
        print("Creating item mapping...")

        all_items = sorted(
            list(set(itertools.chain.from_iterable(sessions_df["items"])))
        )

        item_to_id = {item: idx + 1 for idx, item in enumerate(all_items)}
        id_to_item = {idx: item for item, idx in item_to_id.items()}

        print("Applying item mapping to sessions (in parallel)...")
        # --- Use parallel_apply instead of apply ---
        sessions_df["items"] = sessions_df["items"].parallel_apply(
            lambda x: [item_to_id[item] for item in x]
        )

        return sessions_df, item_to_id, id_to_item

    def split_data(
        self, sessions_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets based on time."""
        print("Splitting data...")
        sessions_df = sessions_df.sort_values("timestamp").reset_index(drop=True)
        test_start_time = sessions_df["timestamp"].max() - timedelta(
            days=self.test_days
        )
        is_test = sessions_df["timestamp"] >= test_start_time
        test_data = sessions_df[is_test]
        train_val_data = sessions_df[~is_test]
        val_size = int(len(train_val_data) * self.validation_split)
        val_data = train_val_data.tail(val_size)
        train_data = train_val_data.head(len(train_val_data) - val_size)
        return train_data, val_data, test_data

    def create_global_graph_data(
        self, train_sessions: pd.DataFrame
    ) -> Tuple[Dict, Dict]:
        """Create co-occurrence and transition graphs from training data in parallel."""
        print("Creating global graph data (in parallel)...")

        # Use joblib to process each session's graph contribution in parallel
        # n_jobs=-1 means use all available CPU cores
        results = Parallel(n_jobs=-1)(
            delayed(_process_graph_chunk)(items, self.cooccurrence_window)
            for items in tqdm(train_sessions["items"], desc="Dispatching graph jobs")
        )

        # Merge the results from all parallel processes
        final_cooccurrence = defaultdict(lambda: defaultdict(int))
        final_transition = defaultdict(lambda: defaultdict(int))

        for cooc, trans in tqdm(results, desc="Merging graph results"):
            for u, neighbors in cooc.items():
                for v, count in neighbors.items():
                    final_cooccurrence[u][v] += count
            for u, neighbors in trans.items():
                for v, count in neighbors.items():
                    final_transition[u][v] += count

        return dict(final_cooccurrence), dict(final_transition)

    # The rest of the BasePreprocessor methods (save_processed_data, process) remain the same
    # ... (code omitted for brevity, it's identical to the previous version)

    def save_processed_data(self, output_dir: str, **kwargs: Any):
        """Save all processed data artifacts to disk."""
        print(f"Saving processed data to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)

        # Save data splits
        kwargs["train_data"].to_pickle(os.path.join(output_dir, "train.pkl"))
        kwargs["val_data"].to_pickle(os.path.join(output_dir, "validation.pkl"))
        kwargs["test_data"].to_pickle(os.path.join(output_dir, "test.pkl"))

        # Save mappings
        mappings = {
            "item_to_id": kwargs["item_to_id"],
            "id_to_item": kwargs["id_to_item"],
            "num_items": len(kwargs["item_to_id"]),
        }
        with open(os.path.join(output_dir, "item_mappings.pkl"), "wb") as f:
            pickle.dump(mappings, f)

        # Save global graph data
        with open(os.path.join(output_dir, "global_graph_data.pkl"), "wb") as f:
            pickle.dump(kwargs["global_graph_data"], f)

        # Save statistics
        stats = {
            "num_sessions": len(kwargs["train_data"])
            + len(kwargs["val_data"])
            + len(kwargs["test_data"]),
            "num_train_sessions": len(kwargs["train_data"]),
            "num_val_sessions": len(kwargs["val_data"]),
            "num_test_sessions": len(kwargs["test_data"]),
            "num_items": len(kwargs["item_to_id"]),
            "avg_session_length": np.mean(
                [len(s) for s in kwargs["train_data"]["items"]]
            ),
        }
        with open(os.path.join(output_dir, "stats.pkl"), "wb") as f:
            pickle.dump(stats, f)

        print("Data preprocessing completed!")
        print(f"Statistics: {stats}")

    def process(self, input_file: str, output_dir: str):
        """Main preprocessing pipeline."""
        data = self.load_raw_data(input_file)
        sessions_df = self.create_sessions(data)
        sessions_df = self.filter_items(sessions_df)
        sessions_df, item_to_id, id_to_item = self.create_item_mapping(sessions_df)
        train_data, val_data, test_data = self.split_data(sessions_df)
        global_graph_data = self.create_global_graph_data(train_data)

        self.save_processed_data(
            output_dir=output_dir,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            item_to_id=item_to_id,
            id_to_item=id_to_item,
            global_graph_data=global_graph_data,
        )
        return train_data, val_data, test_data


class TmallPreprocessor(BasePreprocessor):
    """Preprocessor for the Tmall dataset."""

    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        print("Loading Tmall raw data...")
        columns = ["user_id", "item_id", "category_id", "behavior_type", "timestamp"]
        data = pd.read_csv(file_path, sep=",", header=None, names=columns)
        data = data[data["behavior_type"] == 0].drop(
            columns=["behavior_type", "category_id"]
        )
        data["timestamp"] = pd.to_datetime(data["timestamp"], unit="s")
        return data

    def create_sessions(self, data: pd.DataFrame) -> pd.DataFrame:
        print("Creating sessions (vectorized)...")
        data = data.sort_values(["user_id", "timestamp"])
        time_diff = data.groupby("user_id")["timestamp"].diff()
        session_boundary = (time_diff > timedelta(minutes=30)) | (
            data["user_id"] != data["user_id"].shift()
        )
        session_ids = session_boundary.cumsum()
        sessions = data.groupby(session_ids).agg(
            user_id=("user_id", "first"),
            items=("item_id", list),
            timestamp=("timestamp", "first"),
            session_len=("item_id", "size"),
        )
        sessions = sessions[
            sessions["session_len"] >= self.min_session_length
        ].reset_index(drop=True)
        return sessions.drop(columns=["session_len"])


class YoochoosePreprocessor(BasePreprocessor):
    """Preprocessor for the Yoochoose dataset."""

    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        print("Loading Yoochoose raw data...")
        columns = ["session_id", "timestamp", "item_id", "category"]
        data = pd.read_csv(
            file_path, header=None, names=columns, dtype={"category": "str"}
        )
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        return data

    def create_sessions(self, data: pd.DataFrame) -> pd.DataFrame:
        print("Creating sessions from Yoochoose data (parallelized)...")
        data = data.sort_values(["session_id", "timestamp"])
        # Use groupby with pandarallel for large datasets
        def session_agg(df):
            return pd.Series({
                "items": list(df["item_id"]),
                "timestamp": df["timestamp"].min(),
                "session_len": len(df["item_id"])
            })
        sessions = data.groupby("session_id").parallel_apply(session_agg)
        sessions = sessions[sessions["session_len"] >= self.min_session_length]
        sessions = sessions.reset_index()
        sessions.rename(columns={"session_id": "user_id"}, inplace=True)
        return sessions[["user_id", "items", "timestamp"]]

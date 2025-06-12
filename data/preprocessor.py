import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
from tqdm import tqdm
import os
from collections import defaultdict


class TmallPreprocessor:
    """Preprocessor for Tmall dataset"""

    def __init__(self, config):
        self.config = config
        self.min_session_length = config["dataset"]["min_session_length"]
        self.min_item_frequency = config["dataset"]["min_item_frequency"]
        self.test_days = config["dataset"]["test_days"]

    def load_raw_data(self, file_path):
        """Load raw Tmall data"""
        print("Loading raw data...")

        # Tmall format: user_id, item_id, category_id, behavior_type, timestamp
        columns = ["user_id", "item_id", "category_id", "behavior_type", "timestamp"]
        data = pd.read_csv(file_path, sep=",", header=None, names=columns)

        # Filter only click behaviors (behavior_type == 0)
        data = data[data["behavior_type"] == 0]

        # Convert timestamp
        data["timestamp"] = pd.to_datetime(data["timestamp"], unit="s")

        return data

    def create_sessions(self, data):
        """Create sessions from user interactions"""
        print("Creating sessions...")

        # Sort by user and timestamp
        data = data.sort_values(["user_id", "timestamp"])

        sessions = []
        session_id = 0

        # Group by user
        for user_id, user_data in tqdm(data.groupby("user_id")):
            user_data = user_data.sort_values("timestamp")

            # Split into sessions (30-minute gap)
            session_start_time = user_data.iloc[0]["timestamp"]
            current_session = []

            for _, row in user_data.iterrows():
                if row["timestamp"] - session_start_time > timedelta(minutes=30):
                    if len(current_session) >= self.min_session_length:
                        sessions.append(
                            {
                                "session_id": session_id,
                                "user_id": user_id,
                                "items": current_session.copy(),
                                "timestamp": session_start_time,
                            }
                        )
                        session_id += 1

                    current_session = [row["item_id"]]
                    session_start_time = row["timestamp"]
                else:
                    current_session.append(row["item_id"])

            # Add last session
            if len(current_session) >= self.min_session_length:
                sessions.append(
                    {
                        "session_id": session_id,
                        "user_id": user_id,
                        "items": current_session,
                        "timestamp": session_start_time,
                    }
                )
                session_id += 1

        return pd.DataFrame(sessions)

    def filter_items(self, sessions_df):
        """Filter items by frequency"""
        print("Filtering items by frequency...")

        # Count item frequencies
        item_counts = defaultdict(int)
        for items in sessions_df["items"]:
            for item in items:
                item_counts[item] += 1

        # Filter items
        valid_items = {
            item
            for item, count in item_counts.items()
            if count >= self.min_item_frequency
        }

        # Filter sessions
        filtered_sessions = []
        for _, row in sessions_df.iterrows():
            filtered_items = [item for item in row["items"] if item in valid_items]
            if len(filtered_items) >= self.min_session_length:
                row["items"] = filtered_items
                filtered_sessions.append(row)

        return pd.DataFrame(filtered_sessions)

    def create_item_mapping(self, sessions_df):
        """Create item ID mapping"""
        print("Creating item mapping...")

        # Get all unique items
        all_items = set()
        for items in sessions_df["items"]:
            all_items.update(items)

        # Create mapping (reserve 0 for padding)
        item_to_id = {item: idx + 1 for idx, item in enumerate(sorted(all_items))}
        id_to_item = {idx: item for item, idx in item_to_id.items()}

        # Apply mapping
        sessions_df["items"] = sessions_df["items"].apply(
            lambda x: [item_to_id[item] for item in x]
        )

        return sessions_df, item_to_id, id_to_item

    def split_data(self, sessions_df):
        """Split data into train/validation/test sets"""
        print("Splitting data...")

        # Sort by timestamp
        sessions_df = sessions_df.sort_values("timestamp")

        # Split by time
        test_start_time = sessions_df["timestamp"].max() - timedelta(
            days=self.test_days
        )

        test_data = sessions_df[sessions_df["timestamp"] >= test_start_time]
        train_val_data = sessions_df[sessions_df["timestamp"] < test_start_time]

        # Split train/validation
        val_size = int(len(train_val_data) * self.config["dataset"]["validation_split"])
        val_data = train_val_data.tail(val_size)
        train_data = train_val_data.head(len(train_val_data) - val_size)

        return train_data, val_data, test_data

    def create_global_graph_data(self, train_sessions):
        """Create data for global graph construction"""
        print("Creating global graph data...")

        # Item co-occurrence within sessions
        cooccurrence = defaultdict(lambda: defaultdict(int))
        transition = defaultdict(lambda: defaultdict(int))

        for items in tqdm(train_sessions["items"]):
            # Co-occurrence
            for i in range(len(items)):
                for j in range(i + 1, min(i + 4, len(items))):  # Window of 3
                    cooccurrence[items[i]][items[j]] += 1
                    cooccurrence[items[j]][items[i]] += 1

            # Transitions
            for i in range(len(items) - 1):
                transition[items[i]][items[i + 1]] += 1

        return dict(cooccurrence), dict(transition)

    def save_processed_data(
        self,
        output_dir,
        train_data,
        val_data,
        test_data,
        item_to_id,
        id_to_item,
        global_graph_data,
    ):
        """Save processed data"""
        print("Saving processed data...")

        os.makedirs(output_dir, exist_ok=True)

        # Save session data
        train_data.to_pickle(os.path.join(output_dir, "train.pkl"))
        val_data.to_pickle(os.path.join(output_dir, "validation.pkl"))
        test_data.to_pickle(os.path.join(output_dir, "test.pkl"))

        # Save mappings
        with open(os.path.join(output_dir, "item_mappings.pkl"), "wb") as f:
            pickle.dump(
                {
                    "item_to_id": item_to_id,
                    "id_to_item": id_to_item,
                    "num_items": len(item_to_id),
                },
                f,
            )

        # Save global graph data
        with open(os.path.join(output_dir, "global_graph_data.pkl"), "wb") as f:
            pickle.dump(global_graph_data, f)

        # Save statistics
        stats = {
            "num_sessions": len(train_data) + len(val_data) + len(test_data),
            "num_train_sessions": len(train_data),
            "num_val_sessions": len(val_data),
            "num_test_sessions": len(test_data),
            "num_items": len(item_to_id),
            "avg_session_length": np.mean([len(s) for s in train_data["items"]]),
        }

        with open(os.path.join(output_dir, "stats.pkl"), "wb") as f:
            pickle.dump(stats, f)

        print(f"Data preprocessing completed!")
        print(f"Statistics: {stats}")

    def process(self, input_file, output_dir):
        """Main preprocessing pipeline"""
        # Load data
        data = self.load_raw_data(input_file)

        # Create sessions
        sessions_df = self.create_sessions(data)

        # Filter items
        sessions_df = self.filter_items(sessions_df)

        # Create item mapping
        sessions_df, item_to_id, id_to_item = self.create_item_mapping(sessions_df)

        # Split data
        train_data, val_data, test_data = self.split_data(sessions_df)

        # Create global graph data
        global_graph_data = self.create_global_graph_data(train_data)

        # Save data
        self.save_processed_data(
            output_dir,
            train_data,
            val_data,
            test_data,
            item_to_id,
            id_to_item,
            global_graph_data,
        )

        return train_data, val_data, test_data


class YoochoosePreprocessor:
    """Preprocessor for the Yoochoose dataset."""

    def __init__(self, config):
        self.config = config
        # Get settings from the config file
        self.min_session_length = config["dataset"]["min_session_length"]
        self.min_item_frequency = config["dataset"]["min_item_frequency"]
        self.test_days = config["dataset"]["test_days"]
        self.validation_split = config["dataset"]["validation_split"]

    def load_raw_data(self, file_path):
        """Load raw Yoochoose data (yoochoose-clicks.dat)"""
        print("Loading raw Yoochoose data...")
        columns = ["session_id", "timestamp", "item_id", "category"]
        data = pd.read_csv(
            file_path, header=None, names=columns, dtype={"category": str}
        )

        # Convert timestamp to datetime objects
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        return data

    def create_sessions(self, data):
        """
        Create sessions from raw data. Yoochoose already has session_id,
        so we just group by it.
        """
        print("Creating sessions...")
        # Sort by session and time to ensure items are in the correct order
        data = data.sort_values(["session_id", "timestamp"])

        sessions = []
        # Group by session_id
        for session_id, session_data in tqdm(data.groupby("session_id")):
            # Ensure the session meets the minimum length requirement
            if len(session_data) >= self.min_session_length:
                sessions.append(
                    {
                        "session_id": session_id,
                        "user_id": session_id,  # Use session_id as user_id for consistency
                        "items": session_data["item_id"].tolist(),
                        "timestamp": session_data["timestamp"].min(),
                    }
                )
        return pd.DataFrame(sessions)

    def filter_items(self, sessions_df):
        """Filter items by frequency (same logic as Tmall)"""
        print("Filtering items by frequency...")
        item_counts = defaultdict(int)
        for items in sessions_df["items"]:
            for item in items:
                item_counts[item] += 1

        valid_items = {
            item
            for item, count in item_counts.items()
            if count >= self.min_item_frequency
        }

        filtered_sessions = []
        for _, row in sessions_df.iterrows():
            filtered_items = [item for item in row["items"] if item in valid_items]
            if len(filtered_items) >= self.min_session_length:
                row["items"] = filtered_items
                filtered_sessions.append(row)
        return pd.DataFrame(filtered_sessions)

    def create_item_mapping(self, sessions_df):
        """Create item ID mapping (same logic as Tmall)"""
        print("Creating item mapping...")
        all_items = set()
        for items in sessions_df["items"]:
            all_items.update(items)

        item_to_id = {item: idx + 1 for idx, item in enumerate(sorted(all_items))}
        id_to_item = {idx: item for item, idx in item_to_id.items()}

        sessions_df["items"] = sessions_df["items"].apply(
            lambda x: [item_to_id[item] for item in x]
        )
        return sessions_df, item_to_id, id_to_item

    def split_data(self, sessions_df):
        """Split data into train/validation/test sets by time (same logic as Tmall)"""
        print("Splitting data...")
        sessions_df = sessions_df.sort_values("timestamp")

        test_start_time = sessions_df["timestamp"].max() - timedelta(
            days=self.test_days
        )
        test_data = sessions_df[sessions_df["timestamp"] >= test_start_time]
        train_val_data = sessions_df[sessions_df["timestamp"] < test_start_time]

        val_size = int(len(train_val_data) * self.validation_split)
        val_data = train_val_data.tail(val_size)
        train_data = train_val_data.head(len(train_val_data) - val_size)
        return train_data, val_data, test_data

    # The rest of the methods are identical to TmallPreprocessor and can be reused.
    # For clarity, we include them here.
    def create_global_graph_data(self, train_sessions):
        """Create data for global graph construction"""
        print("Creating global graph data...")
        cooccurrence = defaultdict(lambda: defaultdict(int))
        transition = defaultdict(lambda: defaultdict(int))
        for items in tqdm(train_sessions["items"]):
            for i in range(len(items)):
                for j in range(i + 1, min(i + 4, len(items))):
                    cooccurrence[items[i]][items[j]] += 1
                    cooccurrence[items[j]][items[i]] += 1
            for i in range(len(items) - 1):
                transition[items[i]][items[i + 1]] += 1
        return dict(cooccurrence), dict(transition)

    def save_processed_data(
        self,
        output_dir,
        train_data,
        val_data,
        test_data,
        item_to_id,
        id_to_item,
        global_graph_data,
    ):
        """Save processed data"""
        print("Saving processed data...")
        os.makedirs(output_dir, exist_ok=True)
        train_data.to_pickle(os.path.join(output_dir, "train.pkl"))
        val_data.to_pickle(os.path.join(output_dir, "validation.pkl"))
        test_data.to_pickle(os.path.join(output_dir, "test.pkl"))
        with open(os.path.join(output_dir, "item_mappings.pkl"), "wb") as f:
            pickle.dump(
                {
                    "item_to_id": item_to_id,
                    "id_to_item": id_to_item,
                    "num_items": len(item_to_id),
                },
                f,
            )
        with open(os.path.join(output_dir, "global_graph_data.pkl"), "wb") as f:
            pickle.dump(global_graph_data, f)
        stats = {
            "num_sessions": len(train_data) + len(val_data) + len(test_data),
            "num_train_sessions": len(train_data),
            "num_val_sessions": len(val_data),
            "num_test_sessions": len(test_data),
            "num_items": len(item_to_id),
            "avg_session_length": np.mean([len(s) for s in train_data["items"]]),
        }
        with open(os.path.join(output_dir, "stats.pkl"), "wb") as f:
            pickle.dump(stats, f)
        print(f"Data preprocessing completed!\nStatistics: {stats}")

    def process(self, input_file, output_dir):
        """Main preprocessing pipeline"""
        data = self.load_raw_data(input_file)
        sessions_df = self.create_sessions(data)
        sessions_df = self.filter_items(sessions_df)
        sessions_df, item_to_id, id_to_item = self.create_item_mapping(sessions_df)
        train_data, val_data, test_data = self.split_data(sessions_df)
        global_graph_data = self.create_global_graph_data(train_data)
        self.save_processed_data(
            output_dir,
            train_data,
            val_data,
            test_data,
            item_to_id,
            id_to_item,
            global_graph_data,
        )
        return train_data, val_data, test_data

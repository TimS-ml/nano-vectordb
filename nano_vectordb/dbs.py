"""
Core implementation of Nano Vector Database.

This module provides the main vector database functionality including:
- Vector storage and retrieval
- Similarity search using various metrics (cosine)
- Persistence to JSON files
- Multi-tenant support with caching
"""

import os
import json
import base64
import hashlib
from uuid import uuid4
import numpy as np
from typing import TypedDict, Literal, Union, Callable
from dataclasses import dataclass, asdict
import sqlite3
import logging
from logging import getLogger

# Configure logging to show INFO level messages
logging.basicConfig(level=logging.INFO)

# Field name constants used in data dictionaries
f_ID = "__id__"  # Unique identifier field for each data entry
f_VECTOR = "__vector__"  # Vector embedding field
f_METRICS = "__metrics__"  # Similarity metrics/scores field

# Type definitions for better type checking and documentation
Data = TypedDict("Data", {"__id__": str, "__vector__": np.ndarray})  # Individual data entry type
DataBase = TypedDict(
    "DataBase", {"embedding_dim": int, "data": list[Data], "matrix": np.ndarray}
)  # Complete database structure type
Float = np.float32  # Default float type for memory efficiency
ConditionLambda = Callable[[Data], bool]  # Filter function type for conditional queries
logger = getLogger("nano-vectordb")  # Logger instance for this module


def array_to_buffer_string(array: np.ndarray) -> str:
    """
    Convert a NumPy array to a base64-encoded string for storage.

    Args:
        array: NumPy array to convert

    Returns:
        Base64-encoded string representation of the array
    """
    return base64.b64encode(array.tobytes()).decode()


def buffer_string_to_array(base64_str: str, dtype=Float) -> np.ndarray:
    """
    Convert a base64-encoded string back to a NumPy array.

    Args:
        base64_str: Base64-encoded string representation of an array
        dtype: Data type for the reconstructed array (default: Float/np.float32)

    Returns:
        Reconstructed NumPy array
    """
    return np.frombuffer(base64.b64decode(base64_str), dtype=dtype)


def load_storage(file_name) -> Union[DataBase, None]:
    """
    Load vector database from a JSON file.

    Args:
        file_name: Path to the JSON storage file

    Returns:
        DataBase dictionary containing the loaded data, or None if file doesn't exist

    Note:
        The matrix is stored as a base64 string in the file and reconstructed here
    """
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        data = json.load(f)
    # Reconstruct the numpy matrix from base64 string
    data["matrix"] = buffer_string_to_array(data["matrix"]).reshape(
        -1, data["embedding_dim"]
    )
    logger.info(f"Load {data['matrix'].shape} data")
    return data


def hash_ndarray(a: np.ndarray) -> str:
    """
    Generate an MD5 hash of a NumPy array for use as a unique identifier.

    Args:
        a: NumPy array to hash

    Returns:
        Hexadecimal string representation of the MD5 hash
    """
    return hashlib.md5(a.tobytes()).hexdigest()


def normalize(a: np.ndarray) -> np.ndarray:
    """
    Normalize vectors to unit length (L2 normalization).

    This is essential for cosine similarity calculations, as normalized vectors
    allow us to compute cosine similarity using simple dot products.

    Args:
        a: Array of vectors to normalize (can be 1D or 2D)

    Returns:
        Normalized array where each vector has unit length
    """
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


@dataclass
class NanoVectorDB:
    """
    A lightweight vector database for similarity search on embeddings.

    This class provides core vector database functionality including:
    - Adding/updating vectors (upsert operation)
    - Querying by similarity (currently supports cosine similarity)
    - Persistence to JSON files
    - Filtering results with custom conditions

    Attributes:
        embedding_dim: Dimensionality of the vectors to be stored
        metric: Similarity metric to use (currently only "cosine" is supported)
        storage_file: Path to the JSON file for persistence
    """
    embedding_dim: int
    metric: Literal["cosine"] = "cosine"
    storage_file: str = "nano-vectordb.json"

    def pre_process(self):
        """
        Pre-process stored vectors based on the selected metric.

        For cosine similarity, this normalizes all vectors to unit length,
        which allows similarity to be computed efficiently using dot products.
        """
        if self.metric == "cosine":
            self.__storage["matrix"] = normalize(self.__storage["matrix"])

    def __post_init__(self):
        """
        Initialize the database after dataclass initialization.

        This method:
        1. Loads existing data from storage file or creates empty storage
        2. Validates embedding dimensions match
        3. Sets up available metrics
        4. Pre-processes vectors based on the selected metric
        """
        # Create default empty storage structure
        default_storage = {
            "embedding_dim": self.embedding_dim,
            "data": [],
            "matrix": np.array([], dtype=Float).reshape(0, self.embedding_dim),
        }
        # Load from file if exists, otherwise use default
        storage: DataBase = load_storage(self.storage_file) or default_storage
        # Ensure embedding dimensions match between saved and initialized values
        assert (
            storage["embedding_dim"] == self.embedding_dim
        ), f"Embedding dim mismatch, expected: {self.embedding_dim}, but loaded: {storage['embedding_dim']}"
        self.__storage = storage
        # Map metric names to their query implementation functions
        self.usable_metrics = {
            "cosine": self._cosine_query,
        }
        # Validate that the requested metric is supported
        assert self.metric in self.usable_metrics, f"Metric {self.metric} not supported"
        # Pre-process vectors (e.g., normalize for cosine similarity)
        self.pre_process()
        logger.info(f"Init {asdict(self)} {len(self.__storage['data'])} data")

    def get_additional_data(self):
        """
        Retrieve additional metadata stored with the database.

        Returns:
            Dictionary of additional data, or empty dict if none exists
        """
        return self.__storage.get("additional_data", {})

    def store_additional_data(self, **kwargs):
        """
        Store additional metadata with the database.

        This can be used to store configuration, timestamps, or any other
        metadata that should be persisted with the database.

        Args:
            **kwargs: Key-value pairs to store as additional data
        """
        self.__storage["additional_data"] = kwargs

    def upsert(self, datas: list[Data]):
        """
        Insert new vectors or update existing ones (upsert = insert + update).

        If a vector with the same ID already exists, it will be updated.
        If no ID is provided, a hash of the vector is used as the ID.

        Args:
            datas: List of data dictionaries, each containing at least a __vector__ field
                   and optionally an __id__ field and other metadata

        Returns:
            Dictionary with 'update' and 'insert' lists containing the IDs of
            updated and inserted entries respectively
        """
        # Create a dictionary indexed by ID (using vector hash if ID not provided)
        _index_datas = {
            data.get(f_ID, hash_ndarray(data[f_VECTOR])): data for data in datas
        }
        # Normalize vectors if using cosine similarity
        if self.metric == "cosine":
            for v in _index_datas.values():
                v[f_VECTOR] = normalize(v[f_VECTOR])
        report_return = {"update": [], "insert": []}

        # First pass: update existing entries
        for i, already_data in enumerate(self.__storage["data"]):
            if already_data[f_ID] in _index_datas:
                update_d = _index_datas.pop(already_data[f_ID])
                self.__storage["matrix"][i] = update_d[f_VECTOR].astype(Float)
                del update_d[f_VECTOR]  # Remove vector before storing metadata
                self.__storage["data"][i] = update_d
                report_return["update"].append(already_data[f_ID])

        # If all entries were updates, return early
        if len(_index_datas) == 0:
            return report_return

        # Second pass: insert new entries
        report_return["insert"].extend(list(_index_datas.keys()))
        new_matrix = np.array(
            [data[f_VECTOR] for data in _index_datas.values()], dtype=Float
        )
        new_datas = []
        for new_k, new_d in _index_datas.items():
            del new_d[f_VECTOR]  # Remove vector before storing metadata
            new_d[f_ID] = new_k
            new_datas.append(new_d)
        self.__storage["data"].extend(new_datas)
        # Vertically stack new vectors with existing matrix
        self.__storage["matrix"] = np.vstack([self.__storage["matrix"], new_matrix])
        return report_return

    def get(self, ids: list[str]):
        """
        Retrieve data entries by their IDs.

        Args:
            ids: List of IDs to retrieve

        Returns:
            List of data dictionaries matching the provided IDs
        """
        return [data for data in self.__storage["data"] if data[f_ID] in ids]

    def delete(self, ids: list[str]):
        """
        Delete data entries by their IDs.

        This removes both the metadata and the corresponding vectors from storage.

        Args:
            ids: List of IDs to delete
        """
        ids = set(ids)
        left_data = []
        delete_index = []
        # Iterate through all data to identify what to keep and what to delete
        for i, data in enumerate(self.__storage["data"]):
            if data[f_ID] in ids:
                delete_index.append(i)
                ids.remove(data[f_ID])
            else:
                left_data.append(data)
        # Update storage with remaining data
        self.__storage["data"] = left_data
        # Remove corresponding vectors from the matrix
        self.__storage["matrix"] = np.delete(
            self.__storage["matrix"], delete_index, axis=0
        )

    def save(self):
        """
        Persist the database to a JSON file.

        The vector matrix is converted to a base64 string for JSON compatibility.
        All metadata and additional data are also saved.
        """
        storage = {
            **self.__storage,
            "matrix": array_to_buffer_string(self.__storage["matrix"]),
        }
        with open(self.storage_file, "w", encoding="utf-8") as f:
            json.dump(storage, f, ensure_ascii=False)

    def __len__(self):
        """
        Get the number of vectors stored in the database.

        Returns:
            Number of stored vectors
        """
        return len(self.__storage["data"])

    def query(
        self,
        query: np.ndarray,
        top_k: int = 10,
        better_than_threshold: float = None,
        filter_lambda: ConditionLambda = None,
    ) -> list[dict]:
        """
        Query the database for similar vectors.

        Args:
            query: Query vector to find similar vectors for
            top_k: Maximum number of results to return (default: 10)
            better_than_threshold: Optional threshold - only return results with
                                   similarity scores better than this value
            filter_lambda: Optional filter function to apply before searching.
                          Should take a Data dict and return True to include it.

        Returns:
            List of data dictionaries sorted by similarity (most similar first),
            each containing the original metadata plus a __metrics__ field with
            the similarity score
        """
        return self.usable_metrics[self.metric](
            query, top_k, better_than_threshold, filter_lambda=filter_lambda
        )

    def _cosine_query(
        self,
        query: np.ndarray,
        top_k: int,
        better_than_threshold: float,
        filter_lambda: ConditionLambda = None,
    ):
        """
        Internal implementation of cosine similarity search.

        This method computes cosine similarity by:
        1. Normalizing the query vector to unit length
        2. Computing dot products with stored vectors (which are already normalized)
        3. Sorting results by similarity score

        Args:
            query: Query vector
            top_k: Maximum number of results to return
            better_than_threshold: Optional minimum similarity score
            filter_lambda: Optional filter to apply before searching

        Returns:
            List of matching data dictionaries with similarity scores
        """
        # Normalize query vector for cosine similarity calculation
        query = normalize(query)

        # Apply filter if provided, otherwise search all vectors
        if filter_lambda is None:
            use_matrix = self.__storage["matrix"]
            filter_index = np.arange(len(self.__storage["data"]))
        else:
            # Build index of vectors that pass the filter
            filter_index = np.array(
                [
                    i
                    for i, data in enumerate(self.__storage["data"])
                    if filter_lambda(data)
                ]
            )
            use_matrix = self.__storage["matrix"][filter_index]

        # Compute cosine similarity via dot product (vectors are pre-normalized)
        scores = np.dot(use_matrix, query)

        # Get indices of top-k highest scores
        sort_index = np.argsort(scores)[-top_k:]
        sort_index = sort_index[::-1]  # Reverse to get descending order

        # Map filtered indices back to absolute indices in original storage
        sort_abs_index = filter_index[sort_index]

        # Build result list with metadata and scores
        results = []
        for abs_i, rel_i in zip(sort_abs_index, sort_index):
            # Skip results below threshold if specified
            if (
                better_than_threshold is not None
                and scores[rel_i] < better_than_threshold
            ):
                break
            results.append({**self.__storage["data"][abs_i], f_METRICS: scores[rel_i]})
        return results


@dataclass
class MultiTenantNanoVDB:
    """
    Multi-tenant vector database with LRU caching.

    This class manages multiple isolated vector databases (tenants), with an
    in-memory cache to avoid loading all tenant databases at once. When the
    cache reaches capacity, least recently used tenants are evicted and saved
    to disk.

    Attributes:
        embedding_dim: Dimensionality of vectors (must be same for all tenants)
        metric: Similarity metric to use (default: "cosine")
        max_capacity: Maximum number of tenant databases to keep in memory
        storage_dir: Directory to store tenant database files
    """
    embedding_dim: int
    metric: Literal["cosine"] = "cosine"
    max_capacity: int = 1000
    storage_dir: str = "./nano_multi_tenant_storage"

    @staticmethod
    def jsonfile_from_id(tenant_id):
        """
        Generate the filename for a tenant's storage file.

        Args:
            tenant_id: Unique identifier for the tenant

        Returns:
            Filename string for the tenant's JSON storage file
        """
        return f"nanovdb_{tenant_id}.json"

    def __post_init__(self):
        """
        Initialize the multi-tenant database manager.

        Validates configuration and sets up empty storage and cache structures.
        """
        if self.max_capacity < 1:
            raise ValueError("max_capacity should be greater than 0")
        # Dictionary of currently loaded tenant databases
        self.__storage: dict[str, NanoVectorDB] = {}
        # Queue tracking access order for LRU eviction
        self.__cache_queue: list[str] = []

    def contain_tenant(self, tenant_id: str) -> bool:
        """
        Check if a tenant exists (either in memory or on disk).

        Args:
            tenant_id: Tenant identifier to check

        Returns:
            True if the tenant exists, False otherwise
        """
        return tenant_id in self.__storage or os.path.exists(
            f"{self.storage_dir}/{self.jsonfile_from_id(tenant_id)}"
        )

    def __load_tenant_in_cache(
        self, tenant_id: str, in_memory_tenant: NanoVectorDB
    ) -> NanoVectorDB:
        """
        Load a tenant database into the in-memory cache.

        If the cache is at capacity, evicts the least recently used tenant
        (first in queue) and saves it to disk before loading the new tenant.

        Args:
            tenant_id: Identifier for the tenant being loaded
            in_memory_tenant: NanoVectorDB instance to cache

        Returns:
            The loaded NanoVectorDB instance
        """
        print(len(self.__storage), self.max_capacity)
        # Evict LRU tenant if cache is full
        if len(self.__storage) >= self.max_capacity:
            vdb = self.__storage.pop(self.__cache_queue.pop(0))
            if not os.path.exists(self.storage_dir):
                os.makedirs(self.storage_dir)
            vdb.save()  # Persist evicted tenant to disk
        # Add new tenant to cache
        self.__storage[tenant_id] = in_memory_tenant
        self.__cache_queue.append(tenant_id)
        pass

    def __load_tenant(self, tenant_id: str) -> NanoVectorDB:
        """
        Load a tenant database from disk or retrieve from cache.

        Args:
            tenant_id: Identifier for the tenant to load

        Returns:
            NanoVectorDB instance for the requested tenant

        Raises:
            ValueError: If the tenant doesn't exist
        """
        # Return from cache if already loaded
        if tenant_id in self.__storage:
            return self.__storage[tenant_id]

        # Validate tenant exists
        if not self.contain_tenant(tenant_id):
            raise ValueError(f"Tenant {tenant_id} not in storage")

        # Load from disk and add to cache
        in_memory_tenant = NanoVectorDB(
            self.embedding_dim,
            metric=self.metric,
            storage_file=f"{self.storage_dir}/{self.jsonfile_from_id(tenant_id)}",
        )
        self.__load_tenant_in_cache(tenant_id, in_memory_tenant)
        return in_memory_tenant

    def create_tenant(self) -> str:
        """
        Create a new tenant database with a unique ID.

        Returns:
            UUID string identifier for the newly created tenant
        """
        tenant_id = str(uuid4())
        in_memory_tenant = NanoVectorDB(
            self.embedding_dim,
            metric=self.metric,
            storage_file=f"{self.storage_dir}/{self.jsonfile_from_id(tenant_id)}",
        )
        self.__load_tenant_in_cache(tenant_id, in_memory_tenant)
        return tenant_id

    def delete_tenant(self, tenant_id: str):
        """
        Permanently delete a tenant database.

        This removes the tenant from cache (if loaded) and deletes its
        storage file from disk.

        Args:
            tenant_id: Identifier of the tenant to delete
        """
        # Remove from cache if loaded
        if tenant_id in self.__storage:
            self.__storage.pop(tenant_id)
            self.__cache_queue.remove(tenant_id)
        # Delete storage file if it exists
        if os.path.exists(f"{self.storage_dir}/{self.jsonfile_from_id(tenant_id)}"):
            os.remove(f"{self.storage_dir}/{self.jsonfile_from_id(tenant_id)}")

    def get_tenant(self, tenant_id: str) -> NanoVectorDB:
        """
        Get a tenant's database instance, loading it if necessary.

        Args:
            tenant_id: Identifier of the tenant to retrieve

        Returns:
            NanoVectorDB instance for the requested tenant

        Raises:
            ValueError: If the tenant doesn't exist
        """
        return self.__load_tenant(tenant_id)

    def save(self):
        """
        Save all currently cached tenant databases to disk.

        This should be called before shutting down to ensure all changes
        are persisted.
        """
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
        for db in self.__storage.values():
            db.save()

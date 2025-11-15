"""
Unit tests for NanoVectorDB functionality.

This module contains comprehensive tests for:
- Basic database initialization and operations
- Vector upsert, query, get, and delete operations
- Conditional filtering
- Additional data storage
- Multi-tenant functionality
"""

import os
import pytest
import numpy as np
from nano_vectordb import NanoVectorDB, MultiTenantNanoVDB
from nano_vectordb.dbs import f_METRICS, f_ID, f_VECTOR


def test_init():
    """
    Test basic initialization, upsertion, and querying of vectors.

    This test:
    1. Creates a database with random vectors
    2. Upserts vectors into the database
    3. Saves and reloads the database
    4. Queries for similar vectors with a threshold
    5. Verifies the query returns the expected most similar vector
    """
    from time import time

    data_len = 1000
    fake_dim = 1024

    # Initialize database
    start = time()
    a = NanoVectorDB(fake_dim)
    print("Load", time() - start)

    # Create random test data
    fake_embeds = np.random.rand(data_len, fake_dim)
    fakes_data = [{f_VECTOR: fake_embeds[i], f_ID: i} for i in range(data_len)]
    query_data = fake_embeds[data_len // 2]  # Use middle vector as query

    # Upsert vectors into database
    start = time()
    r = a.upsert(fakes_data)
    print("Upsert", time() - start)
    a.save()

    # Reload database from disk to test persistence
    a = NanoVectorDB(fake_dim)

    # Query for similar vectors
    start = time()
    r = a.query(query_data, 10, better_than_threshold=0.01)
    # The most similar should be the query vector itself
    assert r[0][f_ID] == data_len // 2
    print(r)
    # Verify result constraints
    assert len(r) <= 10
    for d in r:
        assert d[f_METRICS] >= 0.01  # All results should meet threshold
    # Cleanup
    os.remove("nano-vectordb.json")


def test_same_upsert():
    """
    Test that upserting the same vectors twice updates instead of inserting.

    This verifies that:
    1. First upsert inserts new vectors
    2. Second upsert of same vectors (identified by hash) updates existing entries
    3. The IDs from first insert match the IDs updated in second upsert
    """
    from time import time

    data_len = 1000
    fake_dim = 1024

    start = time()
    a = NanoVectorDB(fake_dim)
    print("Load", time() - start)

    # Create test data without explicit IDs (will use vector hash as ID)
    fake_embeds = np.random.rand(data_len, fake_dim)
    fakes_data = [{f_VECTOR: fake_embeds[i]} for i in range(data_len)]

    # First upsert should insert all vectors
    r1 = a.upsert(fakes_data)
    assert len(r1["insert"]) == len(fakes_data)

    # Second upsert of same vectors should update, not insert
    fakes_data = [{f_VECTOR: fake_embeds[i]} for i in range(data_len)]
    r2 = a.upsert(fakes_data)
    assert r2["update"] == r1["insert"]  # Updated IDs should match inserted IDs


def test_get():
    """
    Test retrieval of vectors by their IDs.

    Verifies that:
    1. Vectors can be inserted with custom metadata
    2. get() retrieves the correct entries by ID
    3. Metadata is preserved and returned correctly
    """
    a = NanoVectorDB(1024)
    # Insert vectors with IDs and additional metadata
    a.upsert(
        [
            {f_VECTOR: np.random.rand(1024), f_ID: str(i), "content": i}
            for i in range(100)
        ]
    )
    # Retrieve specific vectors by ID
    r = a.get(["0", "1", "2"])
    assert len(r) == 3
    # Verify metadata is preserved
    assert r[0]["content"] == 0
    assert r[1]["content"] == 1
    assert r[2]["content"] == 2


def test_delete():
    """
    Test deletion of vectors by their IDs.

    Verifies that:
    1. Vectors can be deleted by ID
    2. Deleted vectors are no longer retrievable
    3. Database length is updated correctly after deletion
    """
    a = NanoVectorDB(1024)
    # Insert 100 vectors
    a.upsert(
        [
            {f_VECTOR: np.random.rand(1024), f_ID: str(i), "content": i}
            for i in range(100)
        ]
    )
    # Delete 3 specific vectors
    a.delete(["0", "50", "90"])

    # Verify deleted vectors cannot be retrieved
    r = a.get(["0", "50", "90"])
    assert len(r) == 0
    # Verify database now has 97 vectors (100 - 3)
    assert len(a) == 97


def test_cond_filter():
    """
    Test conditional filtering during queries.

    Verifies that:
    1. Normal queries return the most similar vector
    2. Filtered queries only search within vectors matching the filter
    3. Filter can restrict search to specific IDs or metadata conditions
    """
    data_len = 10
    fake_dim = 1024

    a = NanoVectorDB(fake_dim)
    fake_embeds = np.random.rand(data_len, fake_dim)
    # Define filter that only allows vectors with ID == 1
    cond_filer = lambda x: x[f_ID] == 1

    fakes_data = [{f_VECTOR: fake_embeds[i], f_ID: i} for i in range(data_len)]
    query_data = fake_embeds[data_len // 2]
    a.upsert(fakes_data)

    assert len(a) == data_len
    # Without filter, should return the query vector itself (ID == 5)
    r = a.query(query_data, 10, better_than_threshold=0.01)
    assert r[0][f_ID] == data_len // 2

    # With filter, should only return vector with ID == 1
    r = a.query(query_data, 10, filter_lambda=cond_filer)
    assert r[0][f_ID] == 1


def test_additonal_data():
    """
    Test storage and retrieval of additional metadata.

    Verifies that:
    1. Additional metadata can be stored with the database
    2. Metadata persists across save/load cycles
    3. Metadata can contain arbitrary key-value pairs
    """
    data_len = 10
    fake_dim = 1024

    a = NanoVectorDB(fake_dim)

    # Store arbitrary metadata
    a.store_additional_data(a=1, b=2, c=3)
    a.save()

    # Reload database and verify metadata persisted
    a = NanoVectorDB(fake_dim)
    assert a.get_additional_data() == {"a": 1, "b": 2, "c": 3}
    # Cleanup
    os.remove("nano-vectordb.json")


def remove_non_empty_dir(dir_path):
    """
    Utility function to remove a directory and all its contents.

    Args:
        dir_path: Path to the directory to remove
    """
    for f in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, f))
    os.rmdir(dir_path)


def test_multi_tenant():
    """
    Test multi-tenant database functionality.

    This comprehensive test verifies:
    1. Validation of max_capacity parameter
    2. Tenant creation with unique IDs
    3. Tenant data persistence across manager reloads
    4. Error handling for non-existent tenants
    5. LRU cache eviction when capacity is reached
    6. Tenant deletion (from cache and disk)
    7. Storage directory creation only when needed
    """
    # Test validation: max_capacity must be > 0
    with pytest.raises(ValueError):
        multi_tenant = MultiTenantNanoVDB(1024, max_capacity=0)

    # Test basic tenant creation and data persistence
    multi_tenant = MultiTenantNanoVDB(1024)
    tenant_id = multi_tenant.create_tenant()
    tenant = multi_tenant.get_tenant(tenant_id)

    # Store data in tenant
    tenant.store_additional_data(a=1, b=2, c=3)
    multi_tenant.save()

    # Reload and verify tenant data persisted
    multi_tenant = MultiTenantNanoVDB(1024)
    assert multi_tenant.contain_tenant(tenant_id)
    tenant = multi_tenant.get_tenant(tenant_id)
    assert tenant.get_additional_data() == {"a": 1, "b": 2, "c": 3}

    # Test error handling for non-existent tenant
    with pytest.raises(ValueError):
        multi_tenant.get_tenant("1")  # not a uuid

    # Test LRU cache eviction with max_capacity=1
    multi_tenant = MultiTenantNanoVDB(1024, max_capacity=1)
    multi_tenant.create_tenant()  # Creates new tenant, evicting the old one
    multi_tenant.get_tenant(tenant_id)  # Load old tenant from disk

    # Test tenant deletion
    multi_tenant.delete_tenant(tenant_id)

    # Verify tenant no longer exists
    multi_tenant = MultiTenantNanoVDB(1024)
    assert not multi_tenant.contain_tenant(tenant_id)
    remove_non_empty_dir("nano_multi_tenant_storage")

    # Test that storage directory is created only when eviction happens
    multi_tenant = MultiTenantNanoVDB(1024, max_capacity=1)
    multi_tenant.create_tenant()  # First tenant stays in cache
    assert not os.path.exists("nano_multi_tenant_storage")
    multi_tenant.create_tenant()  # Second tenant causes eviction, creating directory
    assert os.path.exists("nano_multi_tenant_storage")
    remove_non_empty_dir("nano_multi_tenant_storage")

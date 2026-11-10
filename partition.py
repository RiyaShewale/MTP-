
import numpy as np


def iid_partition(y, n_clients=100, seed=42):
    rng = np.random.default_rng(seed)

    clients = [[] for _ in range(n_clients)]
    start = 0

    # Distribute each class across all clients
    for label in np.unique(y):
        ids = np.flatnonzero(y == label)
        ids = rng.permutation(ids)

        for i, idx in enumerate(ids):
            client_id = (start + i) % n_clients
            clients[client_id].append(idx)

        start = (start + len(ids)) % n_clients

    return {
        i: rng.permutation(clients[i])
        for i in range(n_clients)
    }


def noniid_partition(y, n_clients=100, seed=42):
    rng = np.random.default_rng(seed)

    # Randomize samples within each label
    ids = rng.permutation(len(y))

    # Sort by activity label
    ids = ids[np.argsort(y[ids], kind="stable")]

    # Create 2 shards per client
    n_shards = 2 * n_clients
    shards = np.array_split(ids, n_shards)

    # Randomly assign 2 shards to each client
    order = rng.permutation(n_shards)

    clients = {}

    for i in range(n_clients):
        a = shards[order[2 * i]]
        b = shards[order[2 * i + 1]]

        clients[i] = rng.permutation(
            np.concatenate([a, b])
        )

    return clients


import numpy as np

from load_data import y_train
from partition import iid_partition, noniid_partition


iid = iid_partition(y_train)
noniid = noniid_partition(y_train)


def check_partition(clients, name):
    print(f"\n{name} distribution")
    print("Total clients:", len(clients))

    all_ids = np.concatenate(list(clients.values()))

    # Every training sample must occur exactly once
    assert np.array_equal(
        np.sort(all_ids),
        np.arange(len(y_train))
    )

    for i in range(5):
        ids = clients[i]
        counts = np.bincount(
            y_train[ids],
            minlength=6
        )

        print(f"Client {i}:")
        print("  Samples:", len(ids))
        print("  Class distribution:", counts)

    print("Partition verified successfully!")


check_partition(iid, "IID")
check_partition(noniid, "Non-IID")

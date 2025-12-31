
from pymilvus import MilvusClient

client = MilvusClient(
        uri="http://127.0.0.1:8080",
    )


client.create_partition(
    collection_name="quick_setup",
    partition_name="partitionA"
)

client.load_partitions(
    collection_name="quick_setup",
    partition_names=["partitionA"]
)

res = client.get_load_state(
    collection_name="quick_setup",
    partition_name="partitionA"
)

print(res)
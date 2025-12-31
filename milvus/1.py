from pymilvus import MilvusClient

client = MilvusClient(
        uri="http://127.0.0.1:8080",
    )

res = client.list_partitions(
    collection_name="quick_setup"
)

print(res)
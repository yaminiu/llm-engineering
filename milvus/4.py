from pymilvus import MilvusClient

client = MilvusClient(
        uri="http://127.0.0.1:8080",
    )

res = client.list_collections()

print(res)

res = client.describe_collection(
    collection_name="quick_setup"
)

print(res)
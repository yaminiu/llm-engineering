from pymilvus import MilvusClient

client = MilvusClient(
        uri="http://127.0.0.1:8080",
    )

query_vector = [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592]

res = client.search(
    collection_name="quick_setup",
    data=[query_vector],
    limit=5,
    filter='color like "red%" and likes > 50',
    output_fields=["color", "likes"]
)

for hits in res:
    print("TopK results:")
    for hit in hits:
        print(hit)
from pymilvus import MilvusClient, DataType


client = MilvusClient(
        uri="http://127.0.0.1:8080",
    )


query_vectors = [
    [0.041732933, 0.013779674, -0.027564144, -0.013061441, 0.009748648]
]

res = client.search(
    collection_name="quick_setup",     # target collection
    data=query_vectors,                # query vectors
    limit=3,                           # number of returned entities
)

print(res)

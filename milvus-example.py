import re

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = MilvusClient("database.db")
if not client.has_collection("holy_bible"):

    text = open("data/pg10.txt", "r").read()
    start = "*** START OF THE PROJECT GUTENBERG EBOOK THE KING JAMES VERSION OF THE BIBLE ***"
    end = "*** END OF THE PROJECT GUTENBERG EBOOK THE KING JAMES VERSION OF THE BIBLE ***"
    body = text.split(start, 1)[1].split(end, 1)[0]

    # Verses are hard-wrapped across lines but separated from each other by a
    # blank line, so splitting on blank lines reconstructs whole verses.
    paragraphs = re.split(r"\n\s*\n", body)
    corpus = [
        re.sub(r"\s+", " ", p).strip()
        for p in paragraphs
        if re.match(r"^\d+:\d+\b", p.strip())  # drop book/chapter title lines, TOC, etc.
    ]

    vectors = model.encode(corpus, batch_size=64, show_progress_bar=True, convert_to_numpy=True).astype("float32")

    client.create_collection(
        collection_name="holy_bible",
        dimension=vectors.shape[1],  # 384 for all-MiniLM-L6-v2
        auto_id=True,
    )
    print("Inserting data into the collection...")
    data = [
        {"vector": vector, "text": verse_text}
        for vector, verse_text in zip(vectors, corpus)
    ]
    batch_size = 1000
    for i in range(0, len(data), batch_size):
        res = client.insert(collection_name="holy_bible", data=data[i : i + batch_size])
    print("Data inserted successfully.")
else:
    print("Collection already exists, using it.")
    client.load_collection("holy_bible")


text_queries = ["King of Israel", "Man who betrayed Jesus"]
vector_queries = model.encode(text_queries, convert_to_numpy=True).astype("float32")

res = client.search(
    collection_name="holy_bible",  # target collection
    data=vector_queries,  # a list of one or more query vectors, supports batch
    limit=5,  # how many results to return (topK)
    output_fields=["vector", "text"],  # what fields to return
)

for r in res:
    print(f"Query: {text_queries[res.index(r)]}")
    for hit in r:
        print(f"Score: {hit.score}, Text: {hit.entity.get('text')}")
    print()
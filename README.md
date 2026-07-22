# Milvus semantic search example

A small example that builds a semantic search index over the King James Bible
using [Milvus](https://milvus.io/) (via its embedded `milvus-lite` backend)
and [sentence-transformers](https://www.sbert.net/) embeddings.

## 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

This installs `pymilvus`, `sentence-transformers` and their dependencies
(including `torch`). The first run of the script will also download the
`all-MiniLM-L6-v2` embedding model (a few hundred MB) from the Hugging Face
Hub, so an internet connection is required at least once.

## 3. Download the input data

The script reads the King James Bible text from `data/pg10.txt`, which is not
committed to the repository. Download it from Project Gutenberg:

```bash
mkdir -p data
curl -o data/pg10.txt https://www.gutenberg.org/cache/epub/10/pg10.txt
```

## 4. Run the script

```bash
python milvus-example.py
```

On the first run it will parse the Bible text into individual verses, embed
them, and store them in a local Milvus database (`database.db/`, created in
the project directory). This step takes a couple of minutes on CPU. Subsequent
runs reuse the existing `database.db` and skip straight to querying.

The script prints the top matching verses for a couple of example natural
language queries (e.g. "King of Israel", "Man who betrayed Jesus").

To rebuild the index from scratch (e.g. after changing the embedding model),
delete the local database first:

```bash
rm -rf database.db
```

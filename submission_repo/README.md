# InnoWing Chatbot Challenge 2026

This is the submission folder for the 2026 InnoWing Chatbot Challenge at hku.  
The project is composed of two pipelines.  

- Data Extraction: I use BeautifulSoup for web scraping to extract text and image url. And then, the image context is extracted by prompting llm to descibe them. Both text and image descriptions are embedded, indexed and stored to a Chroma vector database.
- Retrieval and Response: Given a user Query, the system first determine whether it requires counting or decomposing. Then the system executes both precise and semantic search to retrieve contexts and compose them with the system prompt, which is fed to the llm to generate the final answer.

## Architecture

### Build time (offline, unlimited time)

| Step | Script | Output |
|---|---|---|
| Crawl both sites, extract readable text with section headings, collect image records, infer a year and page type per page | `build/scrape.py` | `data/pages.json`, `data/images.json` |
| Heading-first chunking (each chunk carries its heading as a prefix; character split with overlap only as fallback). Chunk size/overlap tuned by answer-span recall on dev + synthetic verbatim QA pairs | `build/index.py` | `data/chroma` |
| Describe every unique image once with the vision model, cache results, retry transient failures, index descriptions into the same store | `build/images.py` | `data/descriptions.json`, `data/chroma` |

### Answer time (`bot/answer.py`, 30 s per question)

```
questions ──► plan (route: count / compound / simple)
          ──► ONE embedding call for every retrieval string in the batch
          ──► retrieve: lexical exact-match pass + vector search
              · count path: metadata filters (year, host site, artefact
                words), wide fetch, explicit totals beat model counts
              · answer path: evidence blocks labelled per sub-question
          ──► ONE chat call per question, run concurrently
          ──► format check ──► one line per answer on stdout
```

Models via the Azure OpenAI gateway: `gpt-4o-mini` (chat),
`gpt-5-mini` (vision, build time only), `text-embedding-3-small`
(embeddings).

## Running

```bash
cp .env.example .env        # fill in AZURE_OPENAI_KEY
pip install -r requirements.txt

python main.py "your question"
python main.py "question one" "question two"   # batch mode
```

Rebuild the index from scratch:

```bash
python build/scrape.py
python build/index.py
python build/images.py
```

## Repository layout

| Path | What it is |
|---|---|
| `main.py` | Grading entry point (given, unedited) |
| `bot/answer.py` | The chatbot: routing, retrieval, prompting, batching |
| `bot/llm.py`, `bot/store.py` | Gateway client and Chroma wrapper (given) |
| `build/` | Offline pipeline: scraper, indexer, image describer |
| `build/_tune_chunk.py`, `build/_gen_qa.py` | Dev tools: chunk-size tuning harness and synthetic QA generator |
| `data/` | Scraped corpus, image descriptions, Chroma index |

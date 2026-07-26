# Evaluation Corpus

This committed English PDF corpus contains the two UET Machine Learning
lectures supplied by the repository owner for local evaluation. `manifest.json`
pins each file hash, page count, retrieval question, and expected source page.

The offline regression test uses deterministic lexical embeddings so it never
downloads model weights. The local evaluation script reruns the same manifest
with BGE-M3 on CPU.

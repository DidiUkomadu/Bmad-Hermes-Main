# Retrieval pipeline — semantic, keyword (BM25), and hybrid retrieval.
#
# Production retrieval implementation for DocuResearch. Separate from the
# evaluation runner's scaffolding retrieval (evaluation/runner.py).
#
# Architecture:
#   SQLite passages
#         │
#         ├───────────────┐
#         ▼               ▼
#   Semantic retrieval   BM25 retrieval
#         │               │
#         └───────┬───────┘
#                 ▼
#           Hybrid retrieval
#                 │
#                 ▼
#       Document scope constraint
#                 │
#                 ▼
#       Score normalization
#                 │
#                 ▼
#       Deterministic ranking
#                 │
#                 ▼
#   Top-N candidate passages
#                 │
#                 ▼
#     Retrieved results

"""Reranker adapter.

Attempts to use the `flashrank` package if available for ultra-fast reranking.
If `flashrank` is not installed, falls back to a simple embedding-based
cosine-similarity reranker using `sentence-transformers`.
"""
import os
from datetime import datetime
from typing import List
import logging

try:
    from flashrank import Ranker, RerankRequest  # type: ignore
    _HAS_FLASHRANK = True
except Exception:
    _HAS_FLASHRANK = False

try:
    from sentence_transformers import SentenceTransformer
    import torch
    import numpy as np
    _HAS_SENTENCE = True
except Exception:
    _HAS_SENTENCE = False

logging.basicConfig(level=logging.INFO)


class Reranker:
    def __init__(self):
        self.use_flashrank = _HAS_FLASHRANK
        self.use_sentence = _HAS_SENTENCE

        if self.use_flashrank:
            self.ranker = Ranker(model_name=os.getenv("RERANKER_MODEL", "ms-marco-MiniLM-L-12-v2"))
            logging.info("🚀 FlashRank initialized")
        elif self.use_sentence:
            model_name = os.getenv("RERANKER_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
            logging.info(f"⏳ Loading sentence-transformers model '{model_name}' for reranking...")
            self.model = SentenceTransformer(model_name)
            # prefer GPU if available
            if torch.cuda.is_available():
                try:
                    self.model.to("cuda")
                except Exception:
                    pass
            logging.info("✅ SentenceTransformer loaded for reranking")
        else:
            logging.warning("No reranker backend available (flashrank nor sentence-transformers). Reranking disabled.")

    def _rerank_nodes(self, query: str, nodes: list, top_k: int = 5, log: bool = False) -> list:
        if not nodes:
            return []

        if self.use_flashrank:
            passages = [{"id": str(i), "text": node.text, "meta": getattr(node, 'original_doc', {}).metadata if hasattr(node, 'original_doc') else {}} for i, node in enumerate(nodes)]
            rerank_request = RerankRequest(query=query, passages=passages)
            results = self.ranker.rerank(rerank_request)
            final_nodes = []
            for res in results[:top_k]:
                idx = int(res["id"])
                original_node = nodes[idx]
                original_node.score = res.get("score", 0.0)
                final_nodes.append(original_node)
            if log:
                logging.info(f"⚡ FlashRank reranked top {top_k} documents")
            return final_nodes

        if self.use_sentence:
            texts = [node.text for node in nodes]
            # encode query and passages
            device = "cuda" if torch.cuda.is_available() else "cpu"
            q_emb = self.model.encode(query, convert_to_tensor=True, device=device)
            p_embs = self.model.encode(texts, convert_to_tensor=True, device=device)
            # compute cosine similarities
            q_norm = q_emb / (q_emb.norm(dim=0) + 1e-8)
            p_norm = p_embs / (p_embs.norm(dim=1, keepdim=True) + 1e-8)
            sims = (p_norm @ q_norm).cpu().numpy()
            # pair and sort
            pairs = list(zip(nodes, sims))
            pairs_sorted = sorted(pairs, key=lambda x: float(x[1]), reverse=True)
            final_nodes = []
            for i, (node, score) in enumerate(pairs_sorted[:top_k]):
                node.score = float(score)
                final_nodes.append(node)
                if log:
                    logging.debug(f"[RANK {i+1}] score={score:.4f}")
            return final_nodes

        # fallback: return top_k as-is
        return nodes[:top_k]


reranker = Reranker()
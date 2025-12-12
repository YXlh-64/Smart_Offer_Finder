# from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import os
from datetime import datetime
from typing import List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv()


class Reranker:
    def __init__(self):
        self.reranker_model_name = os.getenv(
            "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
        )
        self.reranker = None
        self.use_reranking = True

        start_time = datetime.now()
        self._load_reranker()
        end_time = datetime.now()

        print(
            f"Reranker loaded in {(end_time - start_time).total_seconds():.2f} seconds"
        )

    def _load_reranker(self):
        """Load the reranker model"""

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.reranker_model_name
            )
            self.model.eval()

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            print(f"Using Device: {self.device}")

        except Exception as e:
            print(f"Error loading reranker {e}")
            self.reranker = None

    # def _rerank_nodes(self, query: str, nodes: List, top_k: int = 10) -> List:
    #     """Rerank nodes"""

    #     if not self.reranker or not nodes:
    #         return nodes[:top_k]

    #     try:
    #         start_time = datetime.now()
    #         query_doc_pairs = [[query, node.text] for node in nodes]

    #         rerank_scores = self.reranker.predict(query_doc_pairs)

    #         node_score_pairs = list(zip(nodes, rerank_scores))

    #         reranked_pairs = sorted(node_score_pairs, key=lambda x: x[1], reverse=True)

    #         reranked_scores = []

    #         for node, rerank_score in reranked_pairs[:top_k]:
    #             node.score = float(rerank_score)
    #             reranked_scores.append(node)

    #         print(f"Reranked in: {datetime.now() - start_time} seconds ")
    #         for i, node in enumerate(reranked_scores):
    #             print(
    #                 f"Reranked Node {i + 1}: ID: {node.node_id} with score={node.score:.4f}"
    #             )
    #             print(f"Reranked Content preview: {node.text[:100]}...")
    #             print("---")

    #         return reranked_scores

    #     except Exception as e:
    #         print(f"Error during reranking {e}")
    #         return nodes[:top_k]

    def _rerank_nodes(
        self, query: str, nodes: list, top_k: int = 10, log: bool = False
    ) -> list:
        """Rerank nodes based on relevance to the query."""

        print(
            f"[DEBUG] _rerank_nodes called with query length: {len(query)}, nodes count: {len(nodes)}, top_k: {top_k}"
        )

        if not self.tokenizer or not self.model:
            print("[WARN] No reranker found — returning top_k nodes as-is.")
            return nodes[:top_k]
        if not nodes:
            print("[WARN] Empty node list — returning empty list.")
            return []

        try:
            start_time = datetime.now()
            query_doc_pairs = [
                [query, node.text]
                for node in nodes
                if getattr(node, "text", "").strip()
            ]
            print(
                f"[DEBUG] Created {len(query_doc_pairs)} query-doc pairs for reranking"
            )

            # Tokenize and move tensors to GPU
            print(f"[DEBUG] Tokenizing inputs and moving to device: {self.device}")
            inputs = self.tokenizer(
                query_doc_pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)
            print(f"[DEBUG] Input tensor shape: {inputs['input_ids'].shape}")

            print("[DEBUG] Running model inference...")
            with torch.no_grad():
                outputs = self.model(**inputs, return_dict=True)
                rerank_scores = (
                    outputs.logits.view(-1).float().cpu()
                )  # back to CPU for sorting

            print(f"[DEBUG] Generated {len(rerank_scores)} rerank scores")
            print(
                f"[DEBUG] Score range: min={rerank_scores.min():.4f}, max={rerank_scores.max():.4f}, mean={rerank_scores.mean():.4f}"
            )

            # Combine with nodes
            node_score_pairs = list(zip(nodes, rerank_scores))
            reranked_pairs = sorted(node_score_pairs, key=lambda x: x[1], reverse=True)
            print(f"[DEBUG] Sorted {len(reranked_pairs)} node-score pairs")

            reranked_nodes = []
            for i, (node, score) in enumerate(reranked_pairs[:top_k]):
                node.score = float(score)
                reranked_nodes.append(node)

                if log:
                    print(
                        f"[RANK {i + 1}] Score={node.score:.4f}, Text={node.text[:80]}..."
                    )

            duration = datetime.now() - start_time
            print(
                f"[DEBUG] Reranking completed in {duration.total_seconds():.2f} seconds"
            )
            print(f"[DEBUG] Returning {len(reranked_nodes)} reranked nodes")

            return reranked_nodes

        except Exception as e:
            print(f"[ERROR] Exception during reranking: {e}")
            import traceback

            traceback.print_exc()
            return nodes[:top_k]


reranker = Reranker()

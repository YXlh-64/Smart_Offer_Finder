from flashrank import Ranker, RerankRequest
import os

class Reranker:
    def __init__(self):
        # Ultra-lightweight model (40MB) vs your old one (2GB+)
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
        print("🚀 FlashRank initialized on CPU (Fast!)")

    def _rerank_nodes(self, query: str, nodes: list, top_k: int = 5, log: bool = False) -> list:
        if not nodes:
            return []
        
        # Prepare data structure for FlashRank
        passages = [
            {"id": str(i), "text": node.text, "meta": node.original_doc.metadata} 
            for i, node in enumerate(nodes)
        ]

        # Rerank
        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_request)

        # Map back to original objects
        final_nodes = []
        for res in results[:top_k]:
            idx = int(res["id"])
            original_node = nodes[idx]
            original_node.score = res["score"]
            final_nodes.append(original_node)

        if log:
            print(f"⚡ Reranked top {top_k} documents in ms")

        return final_nodes

reranker = Reranker()
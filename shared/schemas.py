from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

@dataclass
class Chunk:
    """A single chunk of legal text, as produced by 02_knowledge_base_person2.

    This mirrors the JSON objects already being produced by Person 2:
        {
          "id": "قوانين الأحوال الشخصية_0",
          "text": "...",
          "metadata": {
              "department": "civil_affairs" | "tax" | "traffic",
              "law_name": "...",
              "categories": ["..."],
              "chunk_index": 0,
              "source": "Egyptian Legal Corpus"
          }
        }
    """
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def department(self) -> Optional[str]:
        return self.metadata.get("department")

    @property
    def law_name(self) -> Optional[str]:
        return self.metadata.get("law_name")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Chunk":
        return cls(id=d["id"], text=d["text"], metadata=d.get("metadata", {}))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RouterOutput:
    """What the router hands to retrieval: the (rewritten) query plus the
    department it believes the query belongs to."""
    query: str
    department: Optional[str] = None   # "civil_affairs" | "tax" | "traffic" | None (search all)
    confidence: Optional[float] = None



@dataclass
class Query:
    text: str
    top_k: int = 5
    department: Optional[str] = None  # optional metadata filter


@dataclass
class RetrievedChunk:
    """A chunk returned by retrieve(), with score breakdown for debugging /
    evaluation."""
    chunk: Chunk
    score: float                       # final fused score used for ranking
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.chunk.id,
            "text": self.chunk.text,
            "metadata": self.chunk.metadata,
            "score": self.score,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
        }


@dataclass
class RetrievalResult:
    query: str
    results: List[RetrievedChunk] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"query": self.query, "results": [r.to_dict() for r in self.results]}


@dataclass
class Answer:
    query: str
    answer_text: str
    sources: List[str] = field(default_factory=list)   # chunk ids used
    verified: Optional[bool] = None

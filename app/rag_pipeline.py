import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from openai import OpenAI
from rank_bm25 import BM25Okapi

load_dotenv()

# Validate required environment variables
def validate_environment_variables() -> None:
    """Validate that all required environment variables are set"""
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for GPT-4o",
        "QDRANT_URI": "Qdrant Cloud endpoint URL",
        "QDRANT_API_KEY": "Qdrant Cloud API key"
    }

    missing_vars = []
    for var_name, description in required_vars.items():
        if not os.getenv(var_name):
            missing_vars.append(f"{var_name} ({description})")

    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables:\n" +
            "\n".join(f"  - {var}" for var in missing_vars) +
            "\n\nPlease check your .env file configuration."
        )

# Validate environment on import
validate_environment_variables()

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URI"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Configuration
# Configuration constants
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # FastEmbed model (matches qdrant-ingestion.py)
VECTOR_SIZE = 384  # BGE small model vector size
LLM_MODEL = "gpt-4o"
COLLECTION_NAME = "atlan_docs_enhanced"
TOP_K = 5
SCORE_THRESHOLD = 0.3  # Minimum similarity score for search results
MAX_TOKENS = 1000  # Maximum tokens for OpenAI response
TEMPERATURE = 0.3  # OpenAI temperature for response generation
CLASSIFICATION_TEMPERATURE = 0.1  # Lower temperature for more consistent classification

# Advanced RAG Configuration
ENABLE_QUERY_ENHANCEMENT = False  # Temporarily disabled - query enhancement using GPT-4o
ENABLE_HYBRID_SEARCH = True  # Enable hybrid vector + keyword search
HYBRID_VECTOR_WEIGHT = 1  # Weight for vector search results (0.0-1.0)
HYBRID_KEYWORD_WEIGHT = 0  # Weight for keyword search results (0.0-1.0)

class AtlanRAG:
    def __init__(self) -> None:
        self.openai_client = openai_client
        self.embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
        self.bm25_index = None
        self.document_texts = []
        self.document_metadata = []
        self._initialize_bm25_index()

    def _initialize_bm25_index(self) -> None:
        """Initialize BM25 index from Qdrant documents for hybrid search"""
        if not ENABLE_HYBRID_SEARCH:
            return

        try:
            # Get all documents from Qdrant for BM25 indexing
            scroll_result = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=10000,  # Adjust based on your document count
                with_payload=True
            )

            documents = scroll_result[0]
            self.document_texts = []
            self.document_metadata = []

            for doc in documents:
                text = doc.payload.get("text", "")
                if text.strip():
                    self.document_texts.append(text)
                    self.document_metadata.append({
                        "id": doc.id,
                        "source_url": doc.payload.get("source_url", ""),
                        "title": doc.payload.get("title", ""),
                        "doc_type": doc.payload.get("doc_type", "")
                    })

            # Create BM25 index
            if self.document_texts:
                tokenized_texts = [text.lower().split() for text in self.document_texts]
                self.bm25_index = BM25Okapi(tokenized_texts)
                print(f"Initialized BM25 index with {len(self.document_texts)} documents")

        except Exception as e:
            print(f"Warning: Could not initialize BM25 index: {e}")
            self.bm25_index = None

    def enhance_query(self, query: str) -> str:
        """Enhance user query using GPT-4o for better search results"""
        if not ENABLE_QUERY_ENHANCEMENT:
            return query

        enhancement_prompt = f"""You are an expert at enhancing search queries for technical documentation. Your task is to expand and improve the user's query to find more relevant information in Atlan's documentation.

Original query: "{query}"

Enhance this query by:
1. Expanding technical acronyms (SSO → Single Sign-On, SAML, authentication)
2. Adding relevant synonyms and related terms
3. Including product-specific terminology
4. Making it more specific for technical documentation search

Return only the enhanced query, no explanation:"""

        try:
            response = self.openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a technical documentation search query enhancer. Return only the enhanced query."},
                    {"role": "user", "content": enhancement_prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            enhanced = response.choices[0].message.content.strip()
            print(f"Query enhanced: '{query}' → '{enhanced}'")
            return enhanced
        except Exception as e:
            print(f"Query enhancement failed: {e}, using original query")
            return query

    def keyword_search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """Perform BM25 keyword search"""
        if not self.bm25_index or not ENABLE_HYBRID_SEARCH:
            return []

        try:
            query_tokens = query.lower().split()
            scores = self.bm25_index.get_scores(query_tokens)

            # Get top-k results with scores
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include results with positive scores
                    results.append({
                        "text": self.document_texts[idx],
                        "source_url": self.document_metadata[idx]["source_url"],
                        "title": self.document_metadata[idx]["title"],
                        "doc_type": self.document_metadata[idx]["doc_type"],
                        "score": float(scores[idx]),
                        "search_type": "keyword"
                    })

            return results

        except Exception as e:
            print(f"Keyword search failed: {e}")
            return []

    def merge_and_rerank(self, vector_results: List[Dict], keyword_results: List[Dict]) -> List[Dict]:
        """Merge and rerank vector and keyword search results"""
        if not ENABLE_HYBRID_SEARCH:
            return vector_results

        # Normalize scores for both result sets
        def normalize_scores(results: List[Dict]) -> List[Dict]:
            if not results:
                return results
            max_score = max(r["score"] for r in results)
            min_score = min(r["score"] for r in results)
            if max_score == min_score:
                return results

            normalized = []
            for r in results:
                normalized_score = (r["score"] - min_score) / (max_score - min_score)
                r_copy = r.copy()
                r_copy["normalized_score"] = normalized_score
                normalized.append(r_copy)
            return normalized

        # Normalize both result sets
        vector_results = normalize_scores(vector_results)
        keyword_results = normalize_scores(keyword_results)

        # Combine and deduplicate based on text content
        combined_results = {}

        # Add vector results with weight
        for result in vector_results:
            text_key = result["text"][:100]  # Use first 100 chars as key
            if text_key not in combined_results:
                result["final_score"] = result.get("normalized_score", result["score"]) * HYBRID_VECTOR_WEIGHT
                result["search_types"] = ["vector"]
                combined_results[text_key] = result
            else:
                # Update score if this is a duplicate
                combined_results[text_key]["final_score"] += result.get("normalized_score", result["score"]) * HYBRID_VECTOR_WEIGHT
                combined_results[text_key]["search_types"].append("vector")

        # Add keyword results with weight
        for result in keyword_results:
            text_key = result["text"][:100]
            if text_key not in combined_results:
                result["final_score"] = result.get("normalized_score", result["score"]) * HYBRID_KEYWORD_WEIGHT
                result["search_types"] = ["keyword"]
                combined_results[text_key] = result
            else:
                # Boost score for documents found by both methods
                combined_results[text_key]["final_score"] += result.get("normalized_score", result["score"]) * HYBRID_KEYWORD_WEIGHT
                combined_results[text_key]["search_types"].append("keyword")

        # Sort by final score and return top results
        final_results = sorted(combined_results.values(), key=lambda x: x["final_score"], reverse=True)[:TOP_K]

        return final_results

    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for user query using FastEmbed"""
        try:
            # Generate embedding using FastEmbed
            embeddings = list(self.embedding_model.embed([query]))
            if embeddings:
                return embeddings[0].tolist() if hasattr(embeddings[0], 'tolist') else list(embeddings[0])
            return []
        except (RuntimeError, ValueError, TypeError) as e:
            print(f"Error generating query embedding: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error generating query embedding: {e}")
            return []
    
    def search_documents(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """Search for relevant documents using enhanced hybrid approach"""
        # Step 1: Enhance the query
        enhanced_query = self.enhance_query(query)

        # Step 2: Vector search
        vector_results = self._vector_search(enhanced_query, top_k)

        # Step 3: Keyword search (if hybrid search is enabled)
        keyword_results = []
        if ENABLE_HYBRID_SEARCH:
            keyword_results = self.keyword_search(enhanced_query, top_k)

        # Step 4: Merge and rerank results
        if ENABLE_HYBRID_SEARCH and keyword_results:
            final_results = self.merge_and_rerank(vector_results, keyword_results)
            print(f"Hybrid search: {len(vector_results)} vector + {len(keyword_results)} keyword → {len(final_results)} final")
            return final_results
        else:
            return vector_results

    def _vector_search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """Perform vector search in Qdrant"""
        query_embedding = self.generate_query_embedding(query)

        if not query_embedding:
            return []

        try:
            search_results = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True,
                score_threshold=SCORE_THRESHOLD
            )

            results = []
            for result in search_results:
                results.append({
                    "text": result.payload["text"],
                    "source_url": result.payload["source_url"],
                    "title": result.payload["title"],
                    "doc_type": result.payload["doc_type"],
                    "score": result.score,
                    "search_type": "vector"
                })

            return results

        except (ConnectionError, TimeoutError) as e:
            print(f"Connection error in vector search: {e}")
            return []
        except (ValueError, KeyError) as e:
            print(f"Data error in vector search: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error in vector search: {e}")
            return []
    
    def extract_unique_sources(self, search_results: List[Dict]) -> List[str]:
        """Extract unique source URLs from search results"""
        sources = []
        seen_urls = set()
        
        for result in search_results:
            url = result.get("source_url", "")
            if url and url not in seen_urls:
                sources.append(url)
                seen_urls.add(url)
        
        return sources
    
    def generate_rag_response(self, query: str, context_docs: List[Dict]) -> str:
        """Generate response using retrieved context with enhanced search info"""
        if not context_docs:
            return "I couldn't find relevant information in the Atlan documentation to answer your question."

        # Prepare context from retrieved documents with search method info
        context_parts = []
        search_methods = []
        for i, doc in enumerate(context_docs, 1):
            search_info = ""
            if "search_types" in doc:
                methods = ", ".join(doc["search_types"])
                search_info = f" [Found via: {methods}]"
                search_methods.extend(doc["search_types"])
            elif "search_type" in doc:
                search_info = f" [Found via: {doc['search_type']}]"
                search_methods.append(doc["search_type"])

            context_parts.append(f"Context {i}:\nSource: {doc['title']}{search_info}\nContent: {doc['text']}\n")

        context = "\n".join(context_parts)

        # Log search method statistics
        if search_methods:
            method_counts = {}
            for method in search_methods:
                method_counts[method] = method_counts.get(method, 0) + 1
            print(f"Search methods used: {method_counts}")
        
        prompt = f"""You are a helpful assistant that answers questions about Atlan based on the provided documentation context. 

        Use the following context to answer the user's question. Be specific and accurate. If the context doesn't contain enough information to fully answer the question, say so clearly.

        Context:
        {context}

        Question: {query}

        Answer:"""

        try:
            response = self.openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions about Atlan based on the provided documentation context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            return response.choices[0].message.content
        except ConnectionError as e:
            print(f"OpenAI API connection error: {e}")
            return "I'm having trouble connecting to the AI service. Please try again in a moment."
        except ValueError as e:
            print(f"OpenAI API validation error: {e}")
            return "I encountered an issue with your request format. Please try rephrasing your question."
        except Exception as e:
            print(f"Unexpected error generating response: {e}")
            return "I encountered an unexpected error while generating a response. Please try again."
    
    def answer_question(self, query: str) -> Dict[str, Any]:
        """Main RAG pipeline function with enhanced search tracking"""
        # Search for relevant documents
        search_results = self.search_documents(query)

        # Extract unique sources
        sources = self.extract_unique_sources(search_results)

        # Analyze search method usage
        search_methods_used = []
        hybrid_results = False
        for result in search_results:
            if "search_types" in result:
                search_methods_used.extend(result["search_types"])
                hybrid_results = True
            elif "search_type" in result:
                search_methods_used.append(result["search_type"])

        # Generate response
        answer = self.generate_rag_response(query, search_results)

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": len(search_results),
            "search_methods_used": list(set(search_methods_used)),
            "hybrid_search_enabled": ENABLE_HYBRID_SEARCH,
            "query_enhancement_enabled": ENABLE_QUERY_ENHANCEMENT,
            "search_results": search_results  # For debugging
        }

# Classification system
class TicketClassifier:
    def __init__(self) -> None:
        self.openai_client = openai_client
    
    def classify_ticket(self, ticket_subject: str, ticket_body: str) -> Dict[str, Any]:
        """Classify a support ticket"""
        
        classification_prompt = f"""You are an AI assistant that classifies customer support tickets for Atlan, a data catalog platform.

        Analyze the following ticket and provide a classification:

        Subject: {ticket_subject}
        Body: {ticket_body}

        Provide your analysis in the following JSON format:
        {{
        "topic_tags": ["tag1", "tag2"],
        "sentiment": "sentiment_value",
        "priority": "priority_level"
        }}

        Topic Tags (choose relevant ones):
        - How-to: General usage questions
        - Product: Core product features and functionality  
        - Connector: Data source connections and integrations
        - Lineage: Data lineage and dependency tracking
        - API/SDK: Programming interfaces and development tools
        - SSO: Single sign-on and authentication
        - Glossary: Business glossary and term management
        - Best practices: Recommendations and methodologies
        - Sensitive data: Data privacy and security concerns

        Sentiment (choose one):
        - Frustrated: User shows signs of frustration or urgency
        - Curious: User is exploring or learning
        - Angry: User expresses strong dissatisfaction
        - Neutral: Professional, matter-of-fact tone

        Priority (choose one):
        - P0 (High): Critical issues, production blockers, urgent business needs
        - P1 (Medium): Important but not critical, moderate business impact
        - P2 (Low): Nice to have, low business impact, general questions

        Respond with only the JSON object, no additional text."""

        try:
            response = self.openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are an AI assistant that classifies customer support tickets for Atlan, a data catalog platform. Always respond with only valid JSON."},
                    {"role": "user", "content": classification_prompt}
                ],
                max_tokens=500,
                temperature=CLASSIFICATION_TEMPERATURE
            )
            # Parse the JSON response
            classification_text = response.choices[0].message.content.strip()
            
            # Remove any code block markers if present
            if classification_text.startswith("```"):
                classification_text = classification_text.split("```")[1]
                if classification_text.startswith("json"):
                    classification_text = classification_text[4:]
            
            classification = json.loads(classification_text)
            return classification
            
        except ConnectionError as e:
            print(f"OpenAI API connection error during classification: {e}")
            return {
                "topic_tags": ["Connection Error"],
                "sentiment": "Neutral",
                "priority": "P1 (Medium)"
            }
        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON parsing error during classification: {e}")
            return {
                "topic_tags": ["Parsing Error"],
                "sentiment": "Neutral",
                "priority": "P1 (Medium)"
            }
        except Exception as e:
            print(f"Unexpected error classifying ticket: {e}")
            return {
                "topic_tags": ["Unknown"],
                "sentiment": "Neutral",
                "priority": "P1 (Medium)"
            }

def test_rag_pipeline() -> None:
    """Test the RAG pipeline with a sample query"""
    rag = AtlanRAG()
    
    test_query = "How do I connect Snowflake to Atlan?"
    result = rag.answer_question(test_query)
    
    print("Test Query:", test_query)
    print("\nAnswer:", result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"- {source}")
    print(f"\nRetrieved {result['retrieved_chunks']} relevant chunks")

def test_classification() -> None:
    """Test the classification system"""
    classifier = TicketClassifier()
    
    test_ticket = {
        "subject": "Connecting Snowflake to Atlan - required permissions?",
        "body": "Hi team, we're trying to set up our primary Snowflake production database as a new source in Atlan, but the connection keeps failing. We've tried using our standard service account, but it's not working. Our entire BI team is blocked on this integration for a major upcoming project, so it's quite urgent. Could you please provide a definitive list of the exact permissions and credentials needed on the Snowflake side to get this working? Thanks."
    }
    
    classification = classifier.classify_ticket(test_ticket["subject"], test_ticket["body"])
    
    print("Test Ticket Classification:")
    print(f"Subject: {test_ticket['subject']}")
    print(f"Classification: {json.dumps(classification, indent=2)}")

# Integrated Pipeline Class for Streamlit
class RAGPipeline:
    """Integrated pipeline combining classification and RAG functionality"""

    def __init__(self) -> None:
        self.rag = AtlanRAG()
        self.classifier = TicketClassifier()
    
    def classify_ticket(self, content: str) -> Dict[str, Any]:
        """Classify a ticket from combined content (subject + body)"""
        # Split content into subject and body if formatted as "Subject: ...\n\n..."
        if content.startswith("Subject: "):
            lines = content.split("\n", 1)
            subject = lines[0].replace("Subject: ", "").strip()
            body = lines[1].strip() if len(lines) > 1 else ""
        else:
            # Treat entire content as body with empty subject
            subject = ""
            body = content
        
        return self.classifier.classify_ticket(subject, body)
    
    def generate_rag_response(self, query: str) -> Dict[str, Any]:
        """Generate RAG response for a query"""
        return self.rag.answer_question(query)

if __name__ == "__main__":
    print("Testing RAG Pipeline...")
    test_rag_pipeline()
    print("\n" + "="*50 + "\n")
    print("Testing Classification...")
    test_classification()
import os
import json
import logging
from logging.handlers import RotatingFileHandler
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from openai import OpenAI

from memory_manager import get_memory_manager
from validators import validate_query, validate_ticket, sanitize_text
from rate_limiter import get_rate_limiter
from retry_logic import retry_openai_call, retry_qdrant_call, retry_embedding_generation
from cache_manager import get_cache
from metrics import get_metrics_collector, PerformanceTimer
from config_validation import validate_rag_settings, get_warnings_for_settings
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('rag_pipeline.log', maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

# Initialize clients with timeouts
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=30.0,  # 30 second timeout for API calls
    max_retries=2   # Retry failed requests up to 2 times
)
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URI"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=10.0,  # 10 second timeout for Qdrant operations
)

# Configuration
# Configuration constants
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # FastEmbed model (matches qdrant-ingestion.py)
VECTOR_SIZE = 384  # BGE small model vector size
LLM_MODEL = "gpt-4o"
COLLECTION_NAME = "atlan_docs"
TOP_K = 5
SCORE_THRESHOLD = 0.3  # Minimum similarity score for search results
MAX_TOKENS = 1000  # Maximum tokens for OpenAI response
TEMPERATURE = 0.3  # OpenAI temperature for response generation
CLASSIFICATION_TEMPERATURE = 0.1  # Lower temperature for more consistent classification

# Advanced RAG Configuration
ENABLE_QUERY_ENHANCEMENT = False  # Temporarily disabled - query enhancement using GPT-4o

class AtlanRAG:
    def __init__(self, settings: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize RAG pipeline with optional settings.

        Args:
            settings: Optional settings dict. If None, uses global constants.
        """
        self.openai_client = openai_client
        self.embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)

        # Use provided settings or fall back to globals
        self.settings = settings or {
            'top_k': TOP_K,
            'score_threshold': SCORE_THRESHOLD,
            'max_tokens': MAX_TOKENS,
            'temperature': TEMPERATURE,
            'classification_temperature': CLASSIFICATION_TEMPERATURE,
            'llm_model': LLM_MODEL,
            'enable_query_enhancement': ENABLE_QUERY_ENHANCEMENT,
            'collection_name': COLLECTION_NAME
        }

        # Initialize production-ready components
        self.rate_limiter = get_rate_limiter()
        self.cache = get_cache()
        self.metrics = get_metrics_collector()

    def enhance_query(self, query: str) -> str:
        """Enhance user query using GPT-4o for better search results"""
        if not self.settings.get('enable_query_enhancement', False):
            return query

        enhancement_prompt = f"""You are an expert at enhancing search queries for technical documentation. Your task is to expand and improve the user's query to find more relevant information in Atlan's documentation.

Original query: "{query}"

Enhance this query by:
1. Expanding technical acronyms (SSO → Single Sign-On, SAML, authentication)
2. Adding relevant synonyms and related terms
3. Including product-specific terminology
4. Making it more specific for technical documentation search

Return only the enhanced query, no explanation:"""

        # Check rate limit
        allowed, wait_time = self.rate_limiter.check_openai_limit()
        if not allowed:
            logger.warning(f"Rate limit exceeded for query enhancement. Wait {wait_time}s")
            return query  # Fallback to original query

        try:
            # Call OpenAI with retry logic (decorator will handle retries)
            @retry_openai_call
            def _call_openai():
                return self.openai_client.chat.completions.create(
                    model=self.settings.get('llm_model', LLM_MODEL),
                    messages=[
                        {"role": "system", "content": "You are a technical documentation search query enhancer. Return only the enhanced query."},
                        {"role": "user", "content": enhancement_prompt}
                    ],
                    max_tokens=200,
                    temperature=0.1
                )

            response = _call_openai()
            enhanced = response.choices[0].message.content.strip()
            logger.info(f"Query enhanced: '{query}' → '{enhanced}'")
            return enhanced
        except Exception as e:
            logger.warning(f"Query enhancement failed: {e}, using original query")
            return query


    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for user query using FastEmbed with caching"""
        # Check cache first
        cached_embedding = self.cache.get_cached_embedding(query)
        if cached_embedding is not None:
            logger.debug(f"Using cached embedding for query: {query[:50]}...")
            return cached_embedding

        # Generate embedding with retry logic
        @retry_embedding_generation
        def _generate_embedding():
            embeddings = list(self.embedding_model.embed([query]))
            if embeddings:
                return embeddings[0].tolist() if hasattr(embeddings[0], 'tolist') else list(embeddings[0])
            return []

        try:
            embedding = _generate_embedding()
            if embedding:
                # Cache the result
                self.cache.set_cached_embedding(query, embedding)
            return embedding
        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Error generating query embedding: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error generating query embedding: {e}")
            return []
    
    def search_documents(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """Search for relevant documents using vector search"""
        if top_k is None:
            top_k = self.settings.get('top_k', TOP_K)

        # Step 1: Enhance the query (optional)
        enhanced_query = self.enhance_query(query)

        # Step 2: Vector search
        return self._vector_search(enhanced_query, top_k)

    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform vector search in Qdrant with caching and rate limiting"""
        # Check search cache first
        collection = self.settings.get('collection_name', COLLECTION_NAME)
        score_threshold = self.settings.get('score_threshold', SCORE_THRESHOLD)

        cached_results = self.cache.get_cached_search(query, top_k, score_threshold, collection)
        if cached_results is not None:
            logger.debug(f"Using cached search results for query: {query[:50]}...")
            return cached_results

        # Check rate limit
        allowed, wait_time = self.rate_limiter.check_qdrant_limit()
        if not allowed:
            logger.warning(f"Rate limit exceeded for Qdrant search. Wait {wait_time}s")
            return []  # Return empty results

        query_embedding = self.generate_query_embedding(query)

        if not query_embedding:
            return []

        # Perform search with retry logic
        @retry_qdrant_call
        def _search():
            return qdrant_client.search(
                collection_name=collection,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True,
                score_threshold=score_threshold
            )

        try:
            search_results = _search()

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

            # Cache the results
            if results:
                self.cache.set_cached_search(query, top_k, score_threshold, collection, results)

            return results

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Connection error in vector search: {e}")
            return []
        except (ValueError, KeyError) as e:
            logger.error(f"Data error in vector search: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in vector search: {e}")
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
    
    def generate_rag_response(self, query: str, context_docs: List[Dict], session_id: Optional[str] = None) -> str:
        """Generate response using retrieved context with conversation memory"""
        if not context_docs:
            return "I couldn't find relevant information in the Atlan documentation to answer your question."

        # Prepare context from retrieved documents
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            context_parts.append(f"Context {i}:\nSource: {doc['title']}\nContent: {doc['text']}\n")

        context = "\n".join(context_parts)

        # Get conversation history if session_id is provided
        conversation_context = ""
        if session_id:
            memory_manager = get_memory_manager()
            conversation_history = memory_manager.get_conversation_context(session_id, include_last_n=5)
            if conversation_history:
                conversation_context = f"\nPrevious Conversation:\n{conversation_history}\n"

        prompt = f"""You are a helpful assistant that answers questions about Atlan based on the provided documentation context.

        Use the following context to answer the user's question. Be specific and accurate. If the context doesn't contain enough information to fully answer the question, say so clearly.

        If there's previous conversation history, consider it for context but focus primarily on the current question and the documentation context.{conversation_context}

        Documentation Context:
        {context}

        Current Question: {query}

        Answer:"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.settings.get('llm_model', LLM_MODEL),
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions about Atlan based on the provided documentation context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.settings.get('max_tokens', MAX_TOKENS),
                temperature=self.settings.get('temperature', TEMPERATURE)
            )
            return response.choices[0].message.content
        except ConnectionError as e:
            logger.error(f"OpenAI API connection error: {e}")
            return "I'm having trouble connecting to the AI service. Please try again in a moment."
        except ValueError as e:
            logger.error(f"OpenAI API validation error: {e}")
            return "I encountered an issue with your request format. Please try rephrasing your question."
        except Exception as e:
            logger.error(f"Unexpected error generating response: {e}")
            return "I encountered an unexpected error while generating a response. Please try again."
    
    def answer_question(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Main RAG pipeline function with conversation memory, metrics, and rate limiting"""
        # Generate query ID for tracking
        query_id = str(uuid.uuid4())[:8]

        # Start performance timer
        with PerformanceTimer(f"query_{query_id}") as timer:
            # Check overall query rate limit
            allowed, wait_time = self.rate_limiter.check_query_limit()
            if not allowed:
                error_msg = f"Too many requests. Please wait {wait_time} seconds."
                logger.warning(f"Query rate limit exceeded: {error_msg}")
                self.metrics.record_query(
                    query_id=query_id,
                    response_time=0.0,
                    tokens_used=0,
                    search_method="none",
                    num_results=0,
                    error=True,
                    error_type="rate_limit"
                )
                return {
                    "answer": error_msg,
                    "sources": [],
                    "retrieved_chunks": 0,
                    "query_enhancement_enabled": False,
                    "search_results": [],
                    "error": error_msg
                }

            # Validate and sanitize input
            is_valid, error_msg = validate_query(query)
            if not is_valid:
                logger.warning(f"Invalid query rejected: {error_msg}")
                self.metrics.record_query(
                    query_id=query_id,
                    response_time=timer.get_elapsed(),
                    tokens_used=0,
                    search_method="none",
                    num_results=0,
                    error=True,
                    error_type="validation_error"
                )
                return {
                    "answer": f"Invalid query: {error_msg}",
                    "sources": [],
                    "retrieved_chunks": 0,
                    "query_enhancement_enabled": self.settings.get('enable_query_enhancement', ENABLE_QUERY_ENHANCEMENT),
                    "search_results": [],
                    "error": error_msg
                }

            # Sanitize the query
            sanitized_query = sanitize_text(query)

            # Search for relevant documents
            search_results = self.search_documents(sanitized_query)

            # Extract unique sources
            sources = self.extract_unique_sources(search_results)

            # Generate response with conversation context (use sanitized query)
            answer = self.generate_rag_response(sanitized_query, search_results, session_id)

            # Estimate tokens used (rough estimate)
            estimated_tokens = len(sanitized_query.split()) * 1.3 + len(answer.split()) * 1.3 + 500

            # Record successful metrics
            self.metrics.record_query(
                query_id=query_id,
                response_time=timer.get_elapsed(),
                tokens_used=int(estimated_tokens),
                search_method="vector",  # Could be dynamic based on actual search method
                num_results=len(search_results),
                error=False,
                cached=False,  # Could check if results were cached
                model=self.settings.get('llm_model', LLM_MODEL)
            )

            logger.info(f"Query {query_id} completed in {timer.get_elapsed():.2f}s with {len(search_results)} results")

            return {
                "answer": answer,
                "sources": sources,
                "retrieved_chunks": len(search_results),
                "query_enhancement_enabled": self.settings.get('enable_query_enhancement', ENABLE_QUERY_ENHANCEMENT),
                "search_results": search_results,  # For debugging
                "query_id": query_id,
                "response_time": timer.get_elapsed()
            }

# Classification system
class TicketClassifier:
    def __init__(self, settings: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize ticket classifier with optional settings.

        Args:
            settings: Optional settings dict. If None, uses global constants.
        """
        self.openai_client = openai_client
        self.settings = settings or {
            'llm_model': LLM_MODEL,
            'classification_temperature': CLASSIFICATION_TEMPERATURE
        }

        # Initialize production components
        self.rate_limiter = get_rate_limiter()
        self.cache = get_cache()
    
    def classify_ticket(self, ticket_subject: str, ticket_body: str) -> Dict[str, Any]:
        """Classify a support ticket"""
        # Validate and sanitize inputs
        is_valid, error_msg = validate_ticket(ticket_subject, ticket_body)
        if not is_valid:
            logger.warning(f"Invalid ticket rejected: {error_msg}")
            return {
                "topic_tags": ["Invalid Input"],
                "sentiment": "Neutral",
                "priority": "P2 (Low)",
                "error": error_msg
            }

        # Sanitize inputs
        sanitized_subject = sanitize_text(ticket_subject)
        sanitized_body = sanitize_text(ticket_body)

        # Check cache first (combine subject + body for cache key)
        cache_key = f"{sanitized_subject}|{sanitized_body}"
        cached_classification = self.cache.get_cached_classification(cache_key)
        if cached_classification is not None:
            logger.debug(f"Using cached classification for ticket: {sanitized_subject[:50]}...")
            return cached_classification

        # Check rate limit
        allowed, wait_time = self.rate_limiter.check_classification_limit()
        if not allowed:
            logger.warning(f"Classification rate limit exceeded. Wait {wait_time}s")
            return {
                "topic_tags": ["Rate Limit Exceeded"],
                "sentiment": "Neutral",
                "priority": "P1 (Medium)",
                "error": f"Too many requests. Please wait {wait_time} seconds."
            }

        classification_prompt = f"""You are an AI assistant that classifies customer support tickets for Atlan, a data catalog platform.

        Analyze the following ticket and provide a classification:

        Subject: {sanitized_subject}
        Body: {sanitized_body}

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

        # Call OpenAI with retry logic
        @retry_openai_call
        def _classify():
            return self.openai_client.chat.completions.create(
                model=self.settings.get('llm_model', LLM_MODEL),
                messages=[
                    {"role": "system", "content": "You are an AI assistant that classifies customer support tickets for Atlan, a data catalog platform. Always respond with only valid JSON."},
                    {"role": "user", "content": classification_prompt}
                ],
                max_tokens=500,
                temperature=self.settings.get('classification_temperature', CLASSIFICATION_TEMPERATURE)
            )

        try:
            response = _classify()
            # Parse the JSON response
            classification_text = response.choices[0].message.content.strip()
            
            # Remove any code block markers if present
            if classification_text.startswith("```"):
                classification_text = classification_text.split("```")[1]
                if classification_text.startswith("json"):
                    classification_text = classification_text[4:]
            
            classification = json.loads(classification_text)

            # Cache the successful classification
            self.cache.set_cached_classification(cache_key, classification)

            return classification

        except ConnectionError as e:
            logger.error(f"OpenAI API connection error during classification: {e}")
            return {
                "topic_tags": ["Connection Error"],
                "sentiment": "Neutral",
                "priority": "P1 (Medium)"
            }
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"JSON parsing error during classification: {e}")
            return {
                "topic_tags": ["Parsing Error"],
                "sentiment": "Neutral",
                "priority": "P1 (Medium)"
            }
        except Exception as e:
            logger.error(f"Unexpected error classifying ticket: {e}")
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
    """Integrated pipeline combining classification and RAG functionality with conversation memory"""

    def __init__(self) -> None:
        # Store current settings for dynamic updates (thread-safe instance-level)
        self.current_settings = {
            'top_k': TOP_K,
            'score_threshold': SCORE_THRESHOLD,
            'max_tokens': MAX_TOKENS,
            'temperature': TEMPERATURE,
            'classification_temperature': CLASSIFICATION_TEMPERATURE,
            'llm_model': LLM_MODEL,
            'enable_query_enhancement': ENABLE_QUERY_ENHANCEMENT,
            'collection_name': COLLECTION_NAME
        }

        # Initialize components with settings (thread-safe)
        self.rag = AtlanRAG(settings=self.current_settings)
        self.classifier = TicketClassifier(settings=self.current_settings)
        self.memory_manager = get_memory_manager()

    def update_settings(self, new_settings: Dict[str, Any]) -> bool:
        """
        Update pipeline settings dynamically (thread-safe).

        WARNING: This method only updates the instance settings, not global constants.
        Each RAGPipeline instance maintains its own configuration.
        """
        try:
            # Update current settings (thread-safe - instance-level only)
            self.current_settings.update(new_settings)

            # Apply settings to RAG and Classifier components
            self.rag.settings.update(new_settings)
            self.classifier.settings.update(new_settings)

            # Log collection change for visibility
            if 'collection_name' in new_settings:
                logger.info(f"Switched to collection: {new_settings['collection_name']}")

            logger.info(f"Settings updated successfully: {list(new_settings.keys())}")
            return True
        except Exception as e:
            logger.error(f"Error updating settings: {e}")
            return False

    def get_current_settings(self) -> Dict[str, Any]:
        """Get current pipeline settings"""
        return self.current_settings.copy()

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

    def generate_rag_response(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate RAG response for a query with conversation memory"""
        return self.rag.answer_question(query, session_id)

    def add_conversation_turn(self, session_id: str, user_message: str, ai_response: str) -> None:
        """Add a complete conversation turn to memory"""
        self.memory_manager.add_user_message(session_id, user_message)
        self.memory_manager.add_ai_message(session_id, ai_response)

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """Get or create a conversation session"""
        return self.memory_manager.get_or_create_session(session_id)

    def clear_conversation(self, session_id: str) -> bool:
        """Clear conversation history for a session"""
        return self.memory_manager.clear_session(session_id)

    def get_conversation_history(self, session_id: str) -> str:
        """Get formatted conversation history"""
        return self.memory_manager.get_conversation_context(session_id)

    def get_memory_stats(self) -> Dict:
        """Get memory usage statistics"""
        return self.memory_manager.get_memory_stats()

if __name__ == "__main__":
    print("Testing RAG Pipeline...")
    test_rag_pipeline()
    print("\n" + "="*50 + "\n")
    print("Testing Classification...")
    test_classification()
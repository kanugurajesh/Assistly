import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from openai import OpenAI

from memory_manager import get_memory_manager

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
COLLECTION_NAME = "atlan_docs"
TOP_K = 5
SCORE_THRESHOLD = 0.3  # Minimum similarity score for search results
MAX_TOKENS = 1000  # Maximum tokens for OpenAI response
TEMPERATURE = 0.3  # OpenAI temperature for response generation
CLASSIFICATION_TEMPERATURE = 0.1  # Lower temperature for more consistent classification

# Advanced RAG Configuration
ENABLE_QUERY_ENHANCEMENT = False  # Temporarily disabled - query enhancement using GPT-4o

class AtlanRAG:
    def __init__(self) -> None:
        self.openai_client = openai_client
        self.embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)

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
        """Search for relevant documents using vector search"""
        # Step 1: Enhance the query (optional)
        enhanced_query = self.enhance_query(query)

        # Step 2: Vector search
        return self._vector_search(enhanced_query, top_k)

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
    
    def answer_question(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Main RAG pipeline function with conversation memory"""
        # Search for relevant documents
        search_results = self.search_documents(query)

        # Extract unique sources
        sources = self.extract_unique_sources(search_results)

        # Generate response with conversation context
        answer = self.generate_rag_response(query, search_results, session_id)

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": len(search_results),
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
    """Integrated pipeline combining classification and RAG functionality with conversation memory"""

    def __init__(self) -> None:
        self.rag = AtlanRAG()
        self.classifier = TicketClassifier()
        self.memory_manager = get_memory_manager()
        # Store current settings for dynamic updates
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

    def update_settings(self, new_settings: Dict[str, Any]) -> bool:
        """Update pipeline settings dynamically"""
        try:
            # Update current settings
            self.current_settings.update(new_settings)

            # Update global constants (for new instances)
            global TOP_K, SCORE_THRESHOLD, MAX_TOKENS, TEMPERATURE, CLASSIFICATION_TEMPERATURE
            global LLM_MODEL, ENABLE_QUERY_ENHANCEMENT, COLLECTION_NAME

            if 'top_k' in new_settings:
                TOP_K = new_settings['top_k']
            if 'score_threshold' in new_settings:
                SCORE_THRESHOLD = new_settings['score_threshold']
            if 'max_tokens' in new_settings:
                MAX_TOKENS = new_settings['max_tokens']
            if 'temperature' in new_settings:
                TEMPERATURE = new_settings['temperature']
            if 'classification_temperature' in new_settings:
                CLASSIFICATION_TEMPERATURE = new_settings['classification_temperature']
            if 'llm_model' in new_settings:
                LLM_MODEL = new_settings['llm_model']
            if 'enable_query_enhancement' in new_settings:
                ENABLE_QUERY_ENHANCEMENT = new_settings['enable_query_enhancement']
            if 'collection_name' in new_settings:
                COLLECTION_NAME = new_settings['collection_name']
                print(f"Switched to collection: {COLLECTION_NAME}")

            return True
        except Exception as e:
            print(f"Error updating settings: {e}")
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
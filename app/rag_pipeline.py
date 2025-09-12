import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from openai import OpenAI

load_dotenv()

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URI"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Configuration
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # FastEmbed model (matches qdrant-ingestion.py)
VECTOR_SIZE = 384  # BGE small model vector size
LLM_MODEL = "gpt-4o"
COLLECTION_NAME = "atlan_docs"
TOP_K = 5

class AtlanRAG:
    def __init__(self):
        self.openai_client = openai_client
        self.embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for user query using FastEmbed"""
        try:
            # Generate embedding using FastEmbed
            embeddings = list(self.embedding_model.embed([query]))
            if embeddings:
                return embeddings[0].tolist() if hasattr(embeddings[0], 'tolist') else list(embeddings[0])
            return []
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return []
    
    def search_documents(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """Search for relevant documents in Qdrant"""
        query_embedding = self.generate_query_embedding(query)
        
        if not query_embedding:
            return []
        
        try:
            search_results = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True,
                score_threshold=0.3
            )
            
            results = []
            for result in search_results:
                results.append({
                    "text": result.payload["text"],
                    "source_url": result.payload["source_url"],
                    "title": result.payload["title"],
                    "doc_type": result.payload["doc_type"],
                    "score": result.score
                })
            
            return results
            
        except Exception as e:
            print(f"Error searching documents: {e}")
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
        """Generate response using retrieved context"""
        if not context_docs:
            return "I couldn't find relevant information in the Atlan documentation to answer your question."
        
        # Prepare context from retrieved documents
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            context_parts.append(f"Context {i}:\nSource: {doc['title']}\nContent: {doc['text']}\n")
        
        context = "\n".join(context_parts)
        
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
                max_tokens=1000,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I encountered an error while generating a response. Please try again."
    
    def answer_question(self, query: str) -> Dict[str, Any]:
        """Main RAG pipeline function"""
        # Search for relevant documents
        search_results = self.search_documents(query)
        
        # Extract unique sources
        sources = self.extract_unique_sources(search_results)
        
        # Generate response
        answer = self.generate_rag_response(query, search_results)
        
        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": len(search_results),
            "search_results": search_results  # For debugging
        }

# Classification system
class TicketClassifier:
    def __init__(self):
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
                temperature=0.1
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
            
        except Exception as e:
            print(f"Error classifying ticket: {e}")
            return {
                "topic_tags": ["Unknown"],
                "sentiment": "Neutral",
                "priority": "P1 (Medium)"
            }

def test_rag_pipeline():
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

def test_classification():
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
    
    def __init__(self):
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
import json
import time
from typing import List, Dict, Any
from app.rag_pipeline import TicketClassifier, AtlanRAG

def load_sample_tickets() -> List[Dict]:
    """Load sample tickets from JSON file"""
    try:
        with open("app/sample_tickets.json", "r", encoding="utf-8") as f:
            tickets = json.load(f)
        print(f"Loaded {len(tickets)} tickets")
        return tickets
    except FileNotFoundError:
        print("sample_tickets.json not found")
        return []
    except Exception as e:
        print(f"Error loading tickets: {e}")
        return []

def classify_all_tickets(tickets: List[Dict]) -> List[Dict]:
    """Classify all tickets using the AI classifier"""
    classifier = TicketClassifier()
    classified_tickets = []
    
    print("Starting ticket classification...")
    
    for i, ticket in enumerate(tickets):
        try:
            print(f"Classifying ticket {i+1}/{len(tickets)}: {ticket['id']}")
            
            classification = classifier.classify_ticket(
                ticket["subject"], 
                ticket["body"]
            )
            
            classified_ticket = {
                **ticket,  # Original ticket data
                "classification": classification
            }
            
            classified_tickets.append(classified_ticket)
            
            # Rate limiting to avoid hitting API limits
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error classifying ticket {ticket['id']}: {e}")
            # Add a fallback classification
            classified_ticket = {
                **ticket,
                "classification": {
                    "topic_tags": ["Unknown"],
                    "sentiment": "Neutral",
                    "priority": "P1 (Medium)"
                }
            }
            classified_tickets.append(classified_ticket)
    
    print(f"Classification complete. Processed {len(classified_tickets)} tickets.")
    return classified_tickets

def save_classified_tickets(classified_tickets: List[Dict], filename: str = "classified_tickets.json"):
    """Save classified tickets to JSON file"""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(classified_tickets, f, indent=2, ensure_ascii=False)
        print(f"Classified tickets saved to {filename}")
    except Exception as e:
        print(f"Error saving classified tickets: {e}")

def generate_classification_summary(classified_tickets: List[Dict]):
    """Generate a summary of the classification results"""
    print("\n" + "="*50)
    print("CLASSIFICATION SUMMARY")
    print("="*50)
    
    # Count topic tags
    topic_counts = {}
    sentiment_counts = {}
    priority_counts = {}
    
    for ticket in classified_tickets:
        classification = ticket.get("classification", {})
        
        # Count topics
        topics = classification.get("topic_tags", [])
        for topic in topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Count sentiments
        sentiment = classification.get("sentiment", "Unknown")
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        # Count priorities
        priority = classification.get("priority", "Unknown")
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
    
    print(f"Total tickets processed: {len(classified_tickets)}")
    
    print(f"\nTopic Distribution:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {topic}: {count}")
    
    print(f"\nSentiment Distribution:")
    for sentiment, count in sorted(sentiment_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {sentiment}: {count}")
    
    print(f"\nPriority Distribution:")
    for priority, count in sorted(priority_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {priority}: {count}")

def test_rag_responses(classified_tickets: List[Dict], rag: AtlanRAG):
    """Test RAG responses for tickets that should get AI-generated answers"""
    
    # Topics that should get RAG responses
    rag_topics = ["How-to", "Product", "Best practices", "API/SDK", "SSO"]
    
    print("\n" + "="*50)
    print("TESTING RAG RESPONSES")
    print("="*50)
    
    rag_tickets = []
    for ticket in classified_tickets:
        classification = ticket.get("classification", {})
        topics = classification.get("topic_tags", [])
        
        # Check if any topic should get RAG response
        if any(topic in rag_topics for topic in topics):
            rag_tickets.append(ticket)
    
    print(f"Found {len(rag_tickets)} tickets that should get RAG responses")
    
    # Test a few RAG responses
    test_count = min(3, len(rag_tickets))
    for i in range(test_count):
        ticket = rag_tickets[i]
        print(f"\nTesting RAG for ticket: {ticket['id']}")
        print(f"Subject: {ticket['subject']}")
        
        # Generate RAG response
        rag_result = rag.answer_question(ticket['subject'])
        
        print(f"Answer: {rag_result['answer'][:200]}...")
        print(f"Sources: {len(rag_result['sources'])} URLs")
        
        time.sleep(1)  # Rate limiting

def main():
    """Main function to process all tickets"""
    print("🎫 Starting bulk ticket processing...")
    print("="*50)
    
    # Step 1: Load tickets
    tickets = load_sample_tickets()
    if not tickets:
        print("No tickets to process. Exiting.")
        return
    
    # Step 2: Classify all tickets
    classified_tickets = classify_all_tickets(tickets)
    
    # Step 3: Save results
    save_classified_tickets(classified_tickets)
    
    # Step 4: Generate summary
    generate_classification_summary(classified_tickets)
    
    # Step 5: Test RAG responses (optional)
    try:
        rag = AtlanRAG()
        test_rag_responses(classified_tickets, rag)
    except Exception as e:
        print(f"RAG testing failed (this is expected if Qdrant is not set up yet): {e}")
    
    print("\n✅ Bulk processing complete!")
    print("Results saved to 'classified_tickets.json'")

if __name__ == "__main__":
    main()
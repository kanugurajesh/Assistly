import streamlit as st
import sys
import os
import json
import time

# Import from local rag_pipeline module in the same directory
from rag_pipeline import RAGPipeline
import requests

# Page configuration
st.set_page_config(
    page_title="Atlan Customer Support Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e40af, #3b82f6);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
    }
    
    .main-header p {
        color: #e0e7ff;
        margin: 0.5rem 0 0 0;
    }
    
    .metric-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    
    .user-message {
        background: #dbeafe;
        border-left: 4px solid #3b82f6;
    }
    
    .assistant-message {
        background: #f0fdf4;
        border-left: 4px solid #10b981;
    }
    
    .analysis-box {
        background: #fef3c7;
        border: 1px solid #f59e0b;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .sources-box {
        background: #ecfdf5;
        border: 1px solid #10b981;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'classified_tickets' not in st.session_state:
    st.session_state.classified_tickets = None
if 'rag_pipeline' not in st.session_state:
    try:
        st.session_state.rag_pipeline = RAGPipeline()
    except Exception as e:
        st.error(f"Failed to initialize RAG pipeline: {str(e)}")
        st.session_state.rag_pipeline = None

def load_sample_tickets():
    """Load sample tickets from file"""
    try:
        # Look for sample_tickets.json in the same directory as main.py (app/)
        sample_file = os.path.join(os.path.dirname(__file__), 'sample_tickets.json')
        with open(sample_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading sample tickets: {str(e)}")
        return []

def classify_tickets_bulk():
    """Classify all tickets using the RAG pipeline"""
    if st.session_state.rag_pipeline is None:
        st.error("RAG pipeline not initialized")
        return None
        
    tickets = load_sample_tickets()
    if not tickets:
        return None
    
    classified_tickets = []
    summary = {
        'topics': {},
        'sentiments': {},
        'priorities': {}
    }
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticket in enumerate(tickets):
        status_text.text(f'Classifying ticket {i+1} of {len(tickets)}: {ticket.get("id", "Unknown")}')
        
        try:
            # Combine subject and body for classification
            content = f"Subject: {ticket.get('subject', '')}\n\n{ticket.get('body', '')}"
            
            # Classify the ticket
            classification = st.session_state.rag_pipeline.classify_ticket(content)
            
            ticket['classification'] = classification
            classified_tickets.append(ticket)
            
            # Update summary statistics
            for topic in classification.get('topic_tags', []):
                summary['topics'][topic] = summary['topics'].get(topic, 0) + 1
            
            sentiment = classification.get('sentiment', 'Unknown')
            summary['sentiments'][sentiment] = summary['sentiments'].get(sentiment, 0) + 1
            
            priority = classification.get('priority', 'Unknown')
            summary['priorities'][priority] = summary['priorities'].get(priority, 0) + 1
            
        except Exception as e:
            st.warning(f"Failed to classify ticket {ticket.get('id', 'Unknown')}: {str(e)}")
            ticket['classification'] = {
                'topic_tags': ['Unknown'],
                'sentiment': 'Unknown',
                'priority': 'Unknown'
            }
            classified_tickets.append(ticket)
        
        progress_bar.progress((i + 1) / len(tickets))
        
        # Add small delay to avoid rate limiting
        time.sleep(0.1)
    
    progress_bar.empty()
    status_text.empty()
    
    return {
        'tickets': classified_tickets,
        'summary': summary,
        'total_tickets': len(classified_tickets)
    }

def get_topic_color(topic):
    """Get color for topic tag"""
    colors = {
        'How-to': '#8b5cf6',
        'Product': '#3b82f6',
        'Connector': '#10b981',
        'Lineage': '#f59e0b',
        'API/SDK': '#ef4444',
        'SSO': '#ec4899',
        'Glossary': '#06b6d4',
        'Best practices': '#84cc16',
        'Sensitive data': '#f97316'
    }
    return colors.get(topic, '#6b7280')

def get_priority_color(priority):
    """Get color for priority level"""
    if 'P0' in priority or 'High' in priority:
        return '#ef4444'
    elif 'P1' in priority or 'Medium' in priority:
        return '#f59e0b'
    elif 'P2' in priority or 'Low' in priority:
        return '#10b981'
    return '#6b7280'

def get_sentiment_color(sentiment):
    """Get color for sentiment"""
    colors = {
        'Angry': '#ef4444',
        'Frustrated': '#f97316',
        'Curious': '#3b82f6',
        'Neutral': '#6b7280'
    }
    return colors.get(sentiment, '#6b7280')

# Sidebar navigation
st.sidebar.title("🤖 Atlan Support Copilot")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["🏠 Home", "📊 Dashboard", "💬 Chat Agent"]
)

# Main content based on selected page
if page == "🏠 Home":
    # Home page
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Atlan Customer Support Copilot</h1>
        <p>AI-powered support system with intelligent ticket classification and retrieval-augmented generation for instant answers using Atlan's documentation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Bulk Ticket Classification")
        st.write("Automatically classify and analyze 30+ sample support tickets with AI-powered categorization by topic, sentiment, and priority.")
        if st.button("🚀 View Dashboard", type="primary"):
            st.experimental_set_query_params(page="📊 Dashboard")
            st.experimental_rerun()
    
    with col2:
        st.markdown("### 💬 Interactive AI Agent")
        st.write("Submit questions and get intelligent responses powered by RAG using Atlan's documentation, with automatic routing for complex issues.")
        if st.button("🤖 Try Chat Agent", type="primary"):
            st.experimental_set_query_params(page="💬 Chat Agent")
            st.experimental_rerun()
    
    st.markdown("---")
    st.markdown("### ✨ Key Features")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        **🏷️ Smart Classification**  
        Topic tags, sentiment analysis, and priority assignment
        """)
    
    with col2:
        st.markdown("""
        **🔍 RAG Responses**  
        Answers powered by Atlan's documentation
        """)
    
    with col3:
        st.markdown("""
        **📚 Source Citations**  
        All responses include documentation links
        """)
    
    with col4:
        st.markdown("""
        **⚡ Real-time Processing**  
        Instant classification and response generation
        """)

elif page == "📊 Dashboard":
    # Dashboard page
    st.markdown("""
    <div class="main-header">
        <h1>📊 Support Tickets Dashboard</h1>
        <p>AI-powered classification of customer support tickets</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load and classify tickets button
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🔄 Load & Classify All Tickets", type="primary"):
            with st.spinner("Classifying tickets... This may take a few minutes."):
                st.session_state.classified_tickets = classify_tickets_bulk()
    
    with col2:
        if st.session_state.classified_tickets:
            st.metric("Total Tickets", st.session_state.classified_tickets['total_tickets'])
    
    # Display results if available
    if st.session_state.classified_tickets:
        data = st.session_state.classified_tickets
        
        # Summary statistics
        st.markdown("### 📈 Summary Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Top Topics**")
            topics = sorted(data['summary']['topics'].items(), key=lambda x: x[1], reverse=True)
            for topic, count in topics[:5]:
                st.write(f"• {topic}: {count}")
        
        with col2:
            st.markdown("**Sentiment Distribution**")
            sentiments = sorted(data['summary']['sentiments'].items(), key=lambda x: x[1], reverse=True)
            for sentiment, count in sentiments:
                st.write(f"• {sentiment}: {count}")
        
        with col3:
            st.markdown("**Priority Levels**")
            priorities = sorted(data['summary']['priorities'].items(), key=lambda x: x[1], reverse=True)
            for priority, count in priorities:
                st.write(f"• {priority}: {count}")
        
        st.markdown("---")
        
        # Tickets table
        st.markdown(f"### 🎫 Classified Tickets ({len(data['tickets'])})")
        
        # Search filter
        search_term = st.text_input("🔍 Search tickets:", placeholder="Search by ID, subject, or body...")
        
        # Filter tickets based on search
        filtered_tickets = data['tickets']
        if search_term:
            filtered_tickets = [
                ticket for ticket in data['tickets']
                if search_term.lower() in ticket.get('id', '').lower() or
                   search_term.lower() in ticket.get('subject', '').lower() or
                   search_term.lower() in ticket.get('body', '').lower()
            ]
        
        # Display tickets
        for ticket in filtered_tickets:
            with st.expander(f"🎫 {ticket.get('id', 'Unknown')} - {ticket.get('subject', 'No Subject')[:50]}..."):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Subject:**")
                    st.write(ticket.get('subject', 'No subject'))
                    
                    st.markdown("**Body:**")
                    st.write(ticket.get('body', 'No body'))
                
                with col2:
                    classification = ticket.get('classification', {})
                    
                    st.markdown("**Topics:**")
                    for topic in classification.get('topic_tags', []):
                        color = get_topic_color(topic)
                        st.markdown(f'<span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin: 2px;">{topic}</span>', unsafe_allow_html=True)
                    
                    st.markdown("**Sentiment:**")
                    sentiment = classification.get('sentiment', 'Unknown')
                    color = get_sentiment_color(sentiment)
                    st.markdown(f'<span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px;">{sentiment}</span>', unsafe_allow_html=True)
                    
                    st.markdown("**Priority:**")
                    priority = classification.get('priority', 'Unknown')
                    color = get_priority_color(priority)
                    st.markdown(f'<span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px;">{priority}</span>', unsafe_allow_html=True)

elif page == "💬 Chat Agent":
    # Chat page
    st.markdown("""
    <div class="main-header">
        <h1>💬 Interactive AI Agent</h1>
        <p>Ask questions about Atlan and get AI-powered responses</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Toggle for analysis view
    show_analysis = st.checkbox("🔍 Show internal analysis (classification details)", value=True)
    
    # Chat messages
    st.markdown("### 💬 Conversation")
    
    # Display chat messages
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #6b7280;">
            <div style="font-size: 4rem;">🤖</div>
            <h3>Welcome! Ask me anything about Atlan.</h3>
            <p>Try questions like: "How do I connect Snowflake?" or "What is data lineage?"</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Assistant:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Show analysis if enabled and available
                if show_analysis and "classification" in message:
                    classification = message["classification"]
                    
                    st.markdown("""
                    <div class="analysis-box">
                        <h4>🔍 Internal Analysis</h4>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Topics:**")
                        for topic in classification.get('topic_tags', []):
                            color = get_topic_color(topic)
                            st.markdown(f'<span style="background-color: {color}; color: white; padding: 2px 6px; border-radius: 10px; font-size: 11px; margin: 1px;">{topic}</span>', unsafe_allow_html=True)
                        
                        st.markdown("**Sentiment:**")
                        sentiment = classification.get('sentiment', 'Unknown')
                        color = get_sentiment_color(sentiment)
                        st.markdown(f'<span style="background-color: {color}; color: white; padding: 2px 6px; border-radius: 10px; font-size: 11px;">{sentiment}</span>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("**Priority:**")
                        priority = classification.get('priority', 'Unknown')
                        color = get_priority_color(priority)
                        st.markdown(f'<span style="background-color: {color}; color: white; padding: 2px 6px; border-radius: 10px; font-size: 11px;">{priority}</span>', unsafe_allow_html=True)
                        
                        st.markdown("**Response Type:**")
                        response_type = message.get("response_type", "unknown")
                        type_color = "#10b981" if response_type == "rag" else "#f97316"
                        type_text = "🔍 RAG Response" if response_type == "rag" else "🔄 Routed to Team"
                        st.markdown(f'<span style="background-color: {type_color}; color: white; padding: 2px 6px; border-radius: 10px; font-size: 11px;">{type_text}</span>', unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Show sources if available
                if "sources" in message and message["sources"]:
                    st.markdown("""
                    <div class="sources-box">
                        <h4>📚 Sources</h4>
                    """, unsafe_allow_html=True)
                    
                    for source in message["sources"]:
                        st.markdown(f'• <a href="{source}" target="_blank">{source}</a>', unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Input form
    st.markdown("### ✉️ Send a Message")
    
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            subject = st.text_input("Subject (optional):", placeholder="Brief subject line...")
            message = st.text_area("Your question or support ticket:", placeholder="Type your question here...", height=100)
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🚀 Send Message", type="primary", use_container_width=True)
    
    if submitted and message.strip():
        if st.session_state.rag_pipeline is None:
            st.error("RAG pipeline not initialized. Please check your configuration.")
        else:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": f"**Subject:** {subject}\n\n{message}" if subject else message
            })
            
            # Process with RAG pipeline
            with st.spinner("Analyzing and generating response..."):
                try:
                    # Format input for pipeline
                    input_text = f"Subject: {subject}\n\n{message}" if subject else message
                    
                    # Get classification
                    classification = st.session_state.rag_pipeline.classify_ticket(input_text)
                    
                    # Determine response type and generate response
                    rag_topics = ['How-to', 'Product', 'Best practices', 'API/SDK', 'SSO']
                    should_use_rag = any(topic in classification.get('topic_tags', []) for topic in rag_topics)
                    
                    if should_use_rag:
                        # Generate RAG response
                        response_data = st.session_state.rag_pipeline.generate_rag_response(input_text)
                        response_content = response_data.get('answer', 'I apologize, but I could not generate a response at this time.')
                        sources = response_data.get('sources', [])
                        response_type = 'rag'
                    else:
                        # Generate routing response
                        primary_topic = classification.get('topic_tags', ['Unknown'])[0] if classification.get('topic_tags') else 'Unknown'
                        response_content = f"This ticket has been classified as a '{primary_topic}' issue and routed to the appropriate team for specialized assistance. You should receive a response within 24 hours."
                        sources = []
                        response_type = 'routing'
                    
                    # Add assistant response
                    assistant_message = {
                        "role": "assistant",
                        "content": response_content,
                        "classification": classification,
                        "response_type": response_type
                    }
                    
                    if sources:
                        assistant_message["sources"] = sources
                    
                    st.session_state.messages.append(assistant_message)
                    
                    # Rerun to show new messages
                    st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"Error processing message: {str(e)}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "I apologize, but I encountered an error while processing your request. Please try again or contact support if the issue persists."
                    })
                    st.experimental_rerun()
    
    # Sample questions
    st.markdown("### 💡 Try these sample questions:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("How do I connect Snowflake to Atlan? What permissions are required?", key="sample1"):
            st.session_state.sample_message = "How do I connect Snowflake to Atlan? What permissions are required?"
            st.experimental_rerun()
        
        if st.button("What is data lineage and how does Atlan track it automatically?", key="sample2"):
            st.session_state.sample_message = "What is data lineage and how does Atlan track it automatically?"
            st.experimental_rerun()
    
    with col2:
        if st.button("How do I set up SAML SSO with my identity provider?", key="sample3"):
            st.session_state.sample_message = "How do I set up SAML SSO with my identity provider?"
            st.experimental_rerun()
        
        if st.button("How do I use the Python SDK to create assets programmatically?", key="sample4"):
            st.session_state.sample_message = "How do I use the Python SDK to create assets programmatically?"
            st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; font-size: 14px;">
    Built with Streamlit, Gemini AI, and Qdrant Vector Database
</div>
""", unsafe_allow_html=True)
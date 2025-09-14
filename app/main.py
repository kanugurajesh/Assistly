import streamlit as st
import os
import json
import time

# Import from local rag_pipeline module in the same directory
try:
    from rag_pipeline import RAGPipeline
except EnvironmentError as e:
    st.error(f"Configuration Error: {e}")
    st.stop()

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
        background: #2563eb;
        border-left: 4px solid #3b82f6;
        color: white;
    }
    
    .assistant-message {
        background: #059669;
        border-left: 4px solid #10b981;
        color: white;
    }
    
    .analysis-box {
        background: #d97706;
        border: 1px solid #f59e0b;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        color: white;
    }
    
    .sources-box {
        background: #059669;
        border: 1px solid #10b981;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'classified_tickets' not in st.session_state:
    st.session_state.classified_tickets = None
if 'conversation_session_id' not in st.session_state:
    st.session_state.conversation_session_id = None
if 'rag_pipeline' not in st.session_state:
    with st.spinner("Initializing AI pipeline..."):
        try:
            st.session_state.rag_pipeline = RAGPipeline()
            # Initialize conversation session
            if st.session_state.rag_pipeline:
                st.session_state.conversation_session_id = st.session_state.rag_pipeline.get_or_create_session()
            st.success("AI pipeline initialized successfully!")
            time.sleep(1)  # Brief pause to show success message
        except Exception as e:
            st.error(f"Failed to initialize RAG pipeline: {str(e)}")
            st.warning("Some features may be limited. Check your environment configuration.")
            st.session_state.rag_pipeline = None

def load_sample_tickets():
    """Load sample tickets from file"""
    try:
        # Look for sample_tickets.json in the same directory as main.py (app/)
        sample_file = os.path.join(os.path.dirname(__file__), 'sample_tickets.json')
        st.info(f"Loading tickets from: {sample_file}")
        
        if not os.path.exists(sample_file):
            st.error(f"Sample tickets file not found at: {sample_file}")
            return []
            
        with open(sample_file, 'r', encoding='utf-8') as f:
            tickets = json.load(f)
            st.success(f"Successfully loaded {len(tickets)} sample tickets")
            return tickets
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

def generate_routing_message(primary_topic, classified_topics):
    """Generate a specific routing message based on the classified topic"""
    topic_messages = {
        'Connector': "This appears to be a data connector integration issue. Our Connectors team specializes in setting up and troubleshooting data source connections. They will help you with authentication, permissions, and connection configuration.",
        'Lineage': "This question is about data lineage and dependency tracking. Our Data Platform team handles lineage-related inquiries and can assist with lineage configuration, troubleshooting, and best practices.",
        'Glossary': "This relates to business glossary and term management. Our Data Governance team manages glossary features and can help with term definitions, hierarchies, and metadata management.",
        'Sensitive data': "This involves data privacy and security concerns. Our Security team will review this inquiry to ensure proper handling of sensitive data requirements and compliance considerations.",
        'Unknown': "We're analyzing the specifics of your request to route it to the most appropriate specialist team.",
    }

    # Get specific message for the topic, with fallback
    specific_message = topic_messages.get(primary_topic, f"This has been categorized as a '{primary_topic}' inquiry and will be handled by our specialized team.")

    # Create comprehensive routing message
    if len(classified_topics) > 1:
        topics_str = ', '.join(classified_topics[:-1]) + f" and {classified_topics[-1]}"
        return f"Your inquiry covers {topics_str} topics. {specific_message} You should receive a response within 24 hours. Reference ID: #{hash(str(classified_topics)) % 10000:04d}"
    else:
        return f"{specific_message} You should receive a response within 24 hours. Reference ID: #{hash(primary_topic) % 10000:04d}"

def determine_response_type(classification):
    """
    Determine whether to use RAG pipeline or route to team based on classification.

    Args:
        classification: Dictionary containing topic_tags, sentiment, priority from ticket classification

    Returns:
        dict: Contains 'should_use_rag' (bool), 'response_type' (str), 'reason' (str), 'primary_topic' (str)
    """
    # Get RAG topics from session state settings, with fallback to default
    default_rag_topics = ['How-to', 'Product', 'Best practices', 'API/SDK', 'SSO']
    rag_topics = st.session_state.get('rag_settings', {}).get('rag_topics', default_rag_topics)

    # Get classified topics, handle edge cases
    classified_topics = classification.get('topic_tags', [])

    # Handle empty or invalid classification
    if not classified_topics or not isinstance(classified_topics, list):
        print(f"Warning: Invalid or empty classification topics: {classified_topics}")
        return {
            'should_use_rag': False,
            'response_type': 'routing',
            'reason': 'Invalid classification - no topics identified',
            'primary_topic': 'Unknown',
            'classified_topics': [],
            'matched_topics': []
        }

    # Normalize topics for case-insensitive comparison
    normalized_classified = [topic.strip().lower() for topic in classified_topics]
    normalized_rag_topics = [topic.strip().lower() for topic in rag_topics]

    # Find matches
    matched_topics = []
    for classified_topic in classified_topics:
        for rag_topic in rag_topics:
            if classified_topic.strip().lower() == rag_topic.strip().lower():
                matched_topics.append(classified_topic)
                break

    # Determine if we should use RAG
    should_use_rag = len(matched_topics) > 0

    # Get primary topic for routing message
    primary_topic = classified_topics[0] if classified_topics else 'Unknown'

    # Create detailed reason
    if should_use_rag:
        reason = f"Found RAG topics: {', '.join(matched_topics)}"
    else:
        reason = f"No RAG topic match found. Classified as: {', '.join(classified_topics)}"

    # Log decision for debugging
    print(f"Routing Decision: {reason}")
    print(f"Classified topics: {classified_topics}")
    print(f"RAG topics: {rag_topics}")
    print(f"Matched topics: {matched_topics}")

    return {
        'should_use_rag': should_use_rag,
        'response_type': 'rag' if should_use_rag else 'routing',
        'reason': reason,
        'primary_topic': primary_topic,
        'classified_topics': classified_topics,
        'matched_topics': matched_topics
    }

def get_available_collections():
    """Get list of available Qdrant collections"""
    try:
        if st.session_state.rag_pipeline:
            # Try to access Qdrant client from the RAG pipeline
            from rag_pipeline import qdrant_client
            collections = qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            return collection_names, None
        else:
            return [], "RAG pipeline not initialized"
    except Exception as e:
        return [], f"Error connecting to Qdrant: {str(e)}"

def get_collection_info(collection_name):
    """Get information about a specific collection"""
    try:
        if st.session_state.rag_pipeline:
            from rag_pipeline import qdrant_client
            info = qdrant_client.get_collection(collection_name)
            return {
                'points_count': info.points_count,
                'vector_size': info.config.params.vectors.size,
                'distance': info.config.params.vectors.distance.value if hasattr(info.config.params.vectors.distance, 'value') else str(info.config.params.vectors.distance)
            }, None
        else:
            return None, "RAG pipeline not initialized"
    except Exception as e:
        return None, f"Error getting collection info: {str(e)}"

def process_sample_question(sample_text):
    """Process a sample question and add both user message and AI response with memory"""
    st.session_state.messages.append({"role": "user", "content": sample_text})

    if st.session_state.rag_pipeline is not None:
        with st.spinner("Processing your question..."):
            try:
                session_id = st.session_state.conversation_session_id

                classification = st.session_state.rag_pipeline.classify_ticket(sample_text)
                routing_decision = determine_response_type(classification)

                if routing_decision['should_use_rag']:
                    response_data = st.session_state.rag_pipeline.generate_rag_response(sample_text, session_id)
                    response_content = response_data.get('answer', 'I apologize, but I could not generate a response at this time.')
                    sources = response_data.get('sources', [])
                    response_type = 'rag'

                    # Include enhanced search information
                    search_info = {
                        "query_enhancement_enabled": response_data.get('query_enhancement_enabled', False),
                        "hybrid_search_enabled": response_data.get('hybrid_search_enabled', False),
                        "search_methods_used": response_data.get('search_methods_used', []),
                        "retrieved_chunks": response_data.get('retrieved_chunks', 0)
                    }
                else:
                    primary_topic = routing_decision['primary_topic']
                    classified_topics = routing_decision['classified_topics']
                    response_content = generate_routing_message(primary_topic, classified_topics)
                    sources = []
                    response_type = 'routing'
                    search_info = None

                # Add conversation to memory
                if session_id:
                    st.session_state.rag_pipeline.add_conversation_turn(session_id, sample_text, response_content)

                # Add conversation to memory
                if session_id:
                    st.session_state.rag_pipeline.add_conversation_turn(session_id, sample_text, response_content)

                assistant_message = {
                    "role": "assistant",
                    "content": response_content,
                    "classification": classification,
                    "response_type": response_type,
                    "routing_decision": routing_decision
                }

                if sources:
                    assistant_message["sources"] = sources

                if search_info:
                    assistant_message["search_info"] = search_info

                st.session_state.messages.append(assistant_message)
            except Exception as e:
                print(f"Error processing sample question: {str(e)}")
                # Generate more specific error messages
                if "classification" in str(e).lower():
                    error_msg = "I'm having trouble analyzing your question right now. This might be a temporary issue with our classification system. Please try again in a moment."
                elif "rag" in str(e).lower() or "search" in str(e).lower():
                    error_msg = "I encountered an issue searching our documentation. Your question has been logged and our support team will get back to you within 24 hours."
                elif "openai" in str(e).lower() or "api" in str(e).lower():
                    error_msg = "I'm experiencing connectivity issues with our AI service. Please try again shortly, or contact support if the problem persists."
                else:
                    error_msg = "I apologize, but I encountered an unexpected error while processing your request. Please try again or contact support if the issue persists."

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "error": True,
                    "error_details": str(e) if st.session_state.get('rag_settings', {}).get('show_analysis', True) else None
                })
    st.rerun()

# Sidebar navigation
st.sidebar.title("🤖 Atlan Support Copilot")

# Add conversation management section in sidebar
if st.session_state.rag_pipeline and st.session_state.conversation_session_id:
    with st.sidebar.expander("💬 Conversation Management", expanded=False):
        # Memory stats
        try:
            memory_stats = st.session_state.rag_pipeline.get_memory_stats()
            st.write(f"**Active Sessions:** {memory_stats.get('active_sessions', 0)}")
            st.write(f"**Total Messages:** {memory_stats.get('total_messages', 0)}")

            # Current session info
            conversation_history = st.session_state.rag_pipeline.get_conversation_history(st.session_state.conversation_session_id)
            if conversation_history:
                st.write("**Current Conversation:**")
                st.write(f"{len(conversation_history.split('User:')) - 1} exchanges")
            else:
                st.write("**No conversation history**")
        except Exception as e:
            st.write("Memory stats unavailable")

        # Clear conversation button
        if st.button("🗑️ Clear Conversation", help="Clear current conversation history"):
            if st.session_state.rag_pipeline and st.session_state.conversation_session_id:
                success = st.session_state.rag_pipeline.clear_conversation(st.session_state.conversation_session_id)
                if success:
                    st.session_state.messages = []  # Clear UI messages too
                    st.success("Conversation cleared!")
                    st.rerun()

# Handle query params for navigation
query_params = st.query_params
if "page" in query_params:
    current_page = query_params["page"]
else:
    current_page = "🏠 Home"

# Map query param values to display names
page_mapping = {
    "📊 Dashboard": "📊 Dashboard",
    "💬 Chat Agent": "💬 Chat Agent",
    "🏠 Home": "🏠 Home",
    "⚙️ Settings": "⚙️ Settings"
}

# Find the index for the selectbox
page_options = ["🏠 Home", "📊 Dashboard", "💬 Chat Agent", "⚙️ Settings"]
try:
    current_index = page_options.index(current_page)
except ValueError:
    current_index = 0
    current_page = "🏠 Home"

page = st.sidebar.selectbox(
    "Choose a page:",
    page_options,
    index=current_index
)

# Update query params if page selection changed
if page != current_page:
    st.query_params["page"] = page
    st.rerun()

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
            st.query_params["page"] = "📊 Dashboard"
            st.rerun()
    
    with col2:
        st.markdown("### 💬 Interactive AI Agent")
        st.write("Submit questions and get intelligent responses powered by RAG using Atlan's documentation, with automatic routing for complex issues.")
        if st.button("🤖 Try Chat Agent", type="primary"):
            st.query_params["page"] = "💬 Chat Agent"
            st.rerun()
    
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
    
    # Toggle for analysis view - use settings default if available
    default_show_analysis = st.session_state.get('rag_settings', {}).get('show_analysis', True)
    show_analysis = st.checkbox("🔍 Show internal analysis (classification details)", value=default_show_analysis)
    
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
                    routing_decision = message.get("routing_decision", {})

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

                    # Show routing decision details if available
                    if routing_decision:
                        st.markdown("---")
                        st.markdown("**🔀 Routing Decision Details:**")
                        col3, col4 = st.columns(2)

                        with col3:
                            if routing_decision.get('matched_topics'):
                                st.markdown(f"**✅ Matched RAG Topics:** {', '.join(routing_decision['matched_topics'])}")
                            else:
                                st.markdown("**❌ No RAG Topics Matched**")

                            if routing_decision.get('reason'):
                                st.markdown(f"**📝 Reason:** {routing_decision['reason']}")

                        with col4:
                            if routing_decision.get('classified_topics'):
                                topics = routing_decision['classified_topics']
                                st.markdown(f"**🏷️ All Classified Topics:** {', '.join(topics)}")

                            current_rag_topics = st.session_state.get('rag_settings', {}).get('rag_topics', ['How-to', 'Product', 'Best practices', 'API/SDK', 'SSO'])
                            st.markdown(f"**⚙️ Current RAG Topics:** {', '.join(current_rag_topics)}")

                    # Show error details if available
                    if message.get("error") and message.get("error_details"):
                        st.markdown("---")
                        st.markdown("**🔴 Error Details:**")
                        st.code(message["error_details"], language="text")

                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Show enhanced search information if available
                if "search_info" in message:
                    search_info = message["search_info"]
                    st.markdown("""
                    <div style="background: #1e40af; border: 1px solid #3b82f6; border-radius: 0.5rem; padding: 1rem; margin: 0.5rem 0; color: white;">
                        <h4>🔍 Search Enhancement Details</h4>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        if search_info.get("query_enhancement_enabled"):
                            st.markdown("✅ **Query Enhancement**: Active")
                        else:
                            st.markdown("❌ **Query Enhancement**: Disabled")

                        if search_info.get("hybrid_search_enabled"):
                            st.markdown("✅ **Hybrid Search**: Active")
                        else:
                            st.markdown("❌ **Hybrid Search**: Vector Only")

                    with col2:
                        methods = search_info.get("search_methods_used", [])
                        if methods:
                            st.markdown(f"**Search Methods**: {', '.join(methods).title()}")

                        chunks = search_info.get("retrieved_chunks", 0)
                        st.markdown(f"**Retrieved Chunks**: {chunks}")

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
            # Format the user message
            user_message = f"Subject: {subject}\n\n{message}" if subject else message

            # Add user message to UI
            st.session_state.messages.append({
                "role": "user",
                "content": user_message
            })

            # Process with RAG pipeline
            with st.spinner("Analyzing and generating response..."):
                try:
                    session_id = st.session_state.conversation_session_id

                    # Format input for pipeline
                    input_text = f"Subject: {subject}\n\n{message}" if subject else message

                    # Get classification
                    classification = st.session_state.rag_pipeline.classify_ticket(input_text)

                    # Determine response type and generate response
                    routing_decision = determine_response_type(classification)

                    if routing_decision['should_use_rag']:
                        # Generate RAG response with conversation memory
                        response_data = st.session_state.rag_pipeline.generate_rag_response(input_text, session_id)
                        response_content = response_data.get('answer', 'I apologize, but I could not generate a response at this time.')
                        sources = response_data.get('sources', [])
                        response_type = 'rag'

                        # Include enhanced search information
                        search_info = {
                            "query_enhancement_enabled": response_data.get('query_enhancement_enabled', False),
                            "hybrid_search_enabled": response_data.get('hybrid_search_enabled', False),
                            "search_methods_used": response_data.get('search_methods_used', []),
                            "retrieved_chunks": response_data.get('retrieved_chunks', 0)
                        }
                    else:
                        # Generate routing response
                        primary_topic = routing_decision['primary_topic']
                        classified_topics = routing_decision['classified_topics']
                        response_content = generate_routing_message(primary_topic, classified_topics)
                        sources = []
                        response_type = 'routing'
                        search_info = None

                    # Add conversation to memory
                    if session_id:
                        st.session_state.rag_pipeline.add_conversation_turn(session_id, input_text, response_content)

                    # Add assistant response
                    assistant_message = {
                        "role": "assistant",
                        "content": response_content,
                        "classification": classification,
                        "response_type": response_type,
                        "routing_decision": routing_decision
                    }

                    if sources:
                        assistant_message["sources"] = sources

                    if search_info:
                        assistant_message["search_info"] = search_info

                    st.session_state.messages.append(assistant_message)

                except Exception as e:
                    print(f"Error processing chat message: {str(e)}")
                    st.error(f"Error processing message: {str(e)}")
                    # Generate more specific error messages
                    if "classification" in str(e).lower():
                        error_msg = "I'm having trouble analyzing your question right now. This might be a temporary issue with our classification system. Please try again in a moment."
                    elif "rag" in str(e).lower() or "search" in str(e).lower():
                        error_msg = "I encountered an issue searching our documentation. Your question has been logged and our support team will get back to you within 24 hours."
                    elif "openai" in str(e).lower() or "api" in str(e).lower():
                        error_msg = "I'm experiencing connectivity issues with our AI service. Please try again shortly, or contact support if the problem persists."
                    else:
                        error_msg = "I apologize, but I encountered an unexpected error while processing your request. Please try again or contact support if the issue persists."

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "error": True,
                        "error_details": str(e) if st.session_state.get('rag_settings', {}).get('show_analysis', True) else None
                    })

            # Rerun to update the chat display
            st.rerun()
    
    # Sample questions
    st.markdown("### 💡 Try these sample questions:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("How do I connect Snowflake to Atlan? What permissions are required?", key="sample1"):
            process_sample_question("How do I connect Snowflake to Atlan? What permissions are required?")

        if st.button("What is data lineage and how does Atlan track it automatically?", key="sample2"):
            process_sample_question("What is data lineage and how does Atlan track it automatically?")

    with col2:
        if st.button("How do I set up SAML SSO with my identity provider?", key="sample3"):
            process_sample_question("How do I set up SAML SSO with my identity provider?")
        
        if st.button("What is the use of atlan cli?", key="sample4"):
            process_sample_question("What is the use of atlan cli?")

elif page == "⚙️ Settings":
    # Settings page
    st.markdown("""
    <div class="main-header">
        <h1>⚙️ RAG Pipeline Settings</h1>
        <p>Configure RAG pipeline parameters and UI preferences</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize default settings if not in session state
    if 'rag_settings' not in st.session_state:
        st.session_state.rag_settings = {
            # Search Settings
            'top_k': 5,
            'score_threshold': 0.3,
            'hybrid_vector_weight': 1.0,
            'hybrid_keyword_weight': 0.0,
            'collection_name': 'atlan_docs',  # Default collection

            # Model Settings
            'max_tokens': 1000,
            'temperature': 0.3,
            'classification_temperature': 0.1,
            'llm_model': 'gpt-4o',

            # Feature Toggles
            'enable_query_enhancement': False,
            'enable_hybrid_search': True,

            # UI Settings
            'show_analysis': True,

            # Routing Settings
            'rag_topics': ['How-to', 'Product', 'Best practices', 'API/SDK', 'SSO']
        }

    # Quick help section
    with st.expander("❓ Settings Help", expanded=False):
        st.markdown("""
        **🔍 Search Settings**: Control how many documents are retrieved and similarity thresholds
        - **Collection**: Which Qdrant collection to search in
        - **TOP_K**: Number of most relevant documents to find
        - **Score Threshold**: Minimum similarity score to include results
        - **Search Weights**: Balance between vector (semantic) and keyword (exact) search
          - Weights are auto-balanced to sum to 1.0 (e.g., Vector 0.7 → Keyword 0.3)

        **🤖 Model Settings**: Configure the AI model behavior
        - **LLM Model**: Which OpenAI model to use for responses
        - **Temperature**: Higher = more creative, Lower = more consistent
        - **Max Tokens**: Maximum length of AI responses

        **⚡ Features**: Enable/disable advanced RAG capabilities
        - **Hybrid Search**: Combine semantic + keyword search for better results
        - **Query Enhancement**: Use AI to improve search queries

        **🔀 Routing Settings**: Configure which topics use AI vs team routing
        - **RAG Topics**: Topics that get AI-powered responses from documentation
        - **Team Routing**: Topics that get routed to specialized support teams

        **🎨 UI Settings**: Customize the interface appearance
        - **Show Analysis**: Display classification details by default
        """)

    # Create tabs for different setting categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Search Settings", "🤖 Model Settings", "⚡ Features", "🔀 Routing", "🎨 UI Settings"])

    with tab1:
        st.markdown("### Search Parameters")

        # Collection selection section
        st.markdown("#### Qdrant Collection")
        col1, col2 = st.columns([3, 1])

        with col1:
            # Get available collections
            available_collections, error = get_available_collections()

            if error:
                # Fallback to text input if we can't get collections
                st.warning(f"⚠️ {error}")

                with st.expander("🔧 Troubleshooting", expanded=False):
                    st.markdown("""
                    **Common issues:**
                    - Check if RAG pipeline is properly initialized
                    - Verify Qdrant credentials in .env file
                    - Ensure Qdrant service is running and accessible
                    - Check network connectivity to Qdrant endpoint
                    """)

                st.session_state.rag_settings['collection_name'] = st.text_input(
                    "Collection name (manual entry)",
                    value=st.session_state.rag_settings['collection_name'],
                    help="Enter the Qdrant collection name manually"
                )
            else:
                # Use dropdown with available collections
                current_collection = st.session_state.rag_settings['collection_name']

                # Ensure current collection is in the list (in case it's custom)
                if current_collection not in available_collections and current_collection:
                    available_collections.append(current_collection)

                if available_collections:
                    try:
                        current_index = available_collections.index(current_collection) if current_collection in available_collections else 0
                    except (ValueError, IndexError):
                        current_index = 0

                    st.session_state.rag_settings['collection_name'] = st.selectbox(
                        "Select Qdrant collection",
                        options=available_collections,
                        index=current_index,
                        help="Choose from available Qdrant collections"
                    )

                    # Show available collections summary
                    if len(available_collections) > 1:
                        with st.expander("📋 Available Collections Summary", expanded=False):
                            for col_name in available_collections:
                                info, _ = get_collection_info(col_name)
                                if info:
                                    st.write(f"**{col_name}**: {info['points_count']:,} points, {info['vector_size']}D vectors")
                                else:
                                    st.write(f"**{col_name}**: Info unavailable")
                else:
                    st.info("No collections found. Create one using the ingestion script.")

                    with st.expander("💡 How to create collections", expanded=False):
                        st.markdown("""
                        **To create a new collection:**
                        1. Use the ingestion script: `python qdrant_ingestion.py`
                        2. Available options:
                           - `--qdrant-collection your_collection_name`
                           - `--source-url https://docs.atlan.com` (filter by URL)
                           - `--recreate` (delete existing collection)

                        **Example:**
                        ```bash
                        python qdrant_ingestion.py --qdrant-collection my_docs --source-url https://docs.atlan.com
                        ```
                        """)

                    st.session_state.rag_settings['collection_name'] = st.text_input(
                        "Collection name",
                        value=st.session_state.rag_settings['collection_name'],
                        help="Enter the Qdrant collection name"
                    )

        with col2:
            # Refresh collections button
            if st.button("🔄 Refresh", help="Refresh available collections"):
                st.rerun()

        # Show collection info if available
        if st.session_state.rag_settings['collection_name']:
            collection_info, info_error = get_collection_info(st.session_state.rag_settings['collection_name'])
            if collection_info:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Points", collection_info['points_count'])
                with col2:
                    st.metric("Vector Size", collection_info['vector_size'])
                with col3:
                    st.metric("Distance", collection_info['distance'])
            elif info_error:
                st.warning(f"ℹ️ Collection info unavailable: {info_error}")

        st.markdown("#### Retrieval Parameters")
        col1, col2 = st.columns(2)

        with col1:
            st.session_state.rag_settings['top_k'] = st.number_input(
                "Number of search results (TOP_K)",
                min_value=1, max_value=20,
                value=st.session_state.rag_settings['top_k'],
                help="Number of most relevant documents to retrieve"
            )

            st.session_state.rag_settings['score_threshold'] = st.slider(
                "Minimum similarity score threshold",
                min_value=0.0, max_value=1.0,
                value=st.session_state.rag_settings['score_threshold'],
                step=0.1,
                help="Minimum similarity score for including search results"
            )

        with col2:
            # Auto-balancing hybrid search weights
            st.markdown("**Hybrid Search Weights** (Auto-balanced to sum = 1.0)")

            # Vector weight slider with callback
            vector_weight = st.slider(
                "Vector search weight",
                min_value=0.0, max_value=1.0,
                value=st.session_state.rag_settings['hybrid_vector_weight'],
                step=0.1,
                help="Weight given to vector search results (keyword weight auto-adjusts)",
                key="vector_weight_slider"
            )

            # Auto-calculate keyword weight
            keyword_weight = 1.0 - vector_weight

            # Update session state with balanced weights
            st.session_state.rag_settings['hybrid_vector_weight'] = vector_weight
            st.session_state.rag_settings['hybrid_keyword_weight'] = keyword_weight

            # Display keyword weight (read-only)
            st.slider(
                "Keyword search weight (auto-calculated)",
                min_value=0.0, max_value=1.0,
                value=keyword_weight,
                step=0.1,
                help="Automatically calculated as (1.0 - vector weight)",
                disabled=True,
                key="keyword_weight_display"
            )

            # Show current balance
            st.info(f"💡 **Current Balance**: Vector {vector_weight:.1f} + Keyword {keyword_weight:.1f} = {vector_weight + keyword_weight:.1f}")

    with tab2:
        st.markdown("### Language Model Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.session_state.rag_settings['llm_model'] = st.selectbox(
                "LLM Model",
                options=['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo'],
                index=['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo'].index(st.session_state.rag_settings['llm_model']),
                help="OpenAI model for response generation"
            )

            st.session_state.rag_settings['max_tokens'] = st.number_input(
                "Maximum response tokens",
                min_value=100, max_value=4000,
                value=st.session_state.rag_settings['max_tokens'],
                step=100,
                help="Maximum number of tokens in AI responses"
            )

        with col2:
            st.session_state.rag_settings['temperature'] = st.slider(
                "Response temperature",
                min_value=0.0, max_value=2.0,
                value=st.session_state.rag_settings['temperature'],
                step=0.1,
                help="Controls creativity vs consistency (0=deterministic, 2=very creative)"
            )

            st.session_state.rag_settings['classification_temperature'] = st.slider(
                "Classification temperature",
                min_value=0.0, max_value=1.0,
                value=st.session_state.rag_settings['classification_temperature'],
                step=0.1,
                help="Temperature for ticket classification (lower=more consistent)"
            )

    with tab3:
        st.markdown("### Feature Toggles")

        col1, col2 = st.columns(2)

        with col1:
            st.session_state.rag_settings['enable_hybrid_search'] = st.checkbox(
                "Enable hybrid search",
                value=st.session_state.rag_settings['enable_hybrid_search'],
                help="Combine vector and keyword search for better results"
            )

        with col2:
            st.session_state.rag_settings['enable_query_enhancement'] = st.checkbox(
                "Enable query enhancement",
                value=st.session_state.rag_settings['enable_query_enhancement'],
                help="Use GPT-4o to enhance user queries before search"
            )

        # Show current feature status
        st.markdown("### Current Feature Status")
        if st.session_state.rag_pipeline:
            col1, col2 = st.columns(2)
            with col1:
                hybrid_status = "✅ Active" if st.session_state.rag_settings['enable_hybrid_search'] else "❌ Disabled"
                st.info(f"**Hybrid Search**: {hybrid_status}")
            with col2:
                enhancement_status = "✅ Active" if st.session_state.rag_settings['enable_query_enhancement'] else "❌ Disabled"
                st.info(f"**Query Enhancement**: {enhancement_status}")

    with tab4:
        st.markdown("### Response Routing Configuration")

        st.markdown("#### RAG-enabled Topics")
        st.write("Select which topics should trigger AI-powered responses using the RAG pipeline. All other topics will be routed to specialized teams.")

        # Get all available topic options from the classifier
        all_topics = [
            'How-to', 'Product', 'Connector', 'Lineage',
            'API/SDK', 'SSO', 'Glossary', 'Best practices',
            'Sensitive data'
        ]

        # Current RAG topics
        current_rag_topics = st.session_state.rag_settings.get('rag_topics', ['How-to', 'Product', 'Best practices', 'API/SDK', 'SSO'])

        # Multiselect for RAG topics
        selected_topics = st.multiselect(
            "Topics that should use RAG pipeline (AI-powered responses)",
            options=all_topics,
            default=current_rag_topics,
            help="Topics selected here will get AI-powered responses from documentation. Others will be routed to teams."
        )

        # Update settings
        st.session_state.rag_settings['rag_topics'] = selected_topics

        # Show current configuration
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### ✅ RAG-enabled Topics")
            if selected_topics:
                for topic in selected_topics:
                    st.write(f"• **{topic}** → AI Response")
            else:
                st.write("*No topics selected - all will be routed*")

        with col2:
            st.markdown("#### 🔄 Team-routed Topics")
            routed_topics = [topic for topic in all_topics if topic not in selected_topics]
            if routed_topics:
                for topic in routed_topics:
                    st.write(f"• **{topic}** → Team Routing")
            else:
                st.write("*All topics will use AI responses*")

        # Configuration warnings
        if not selected_topics:
            st.warning("⚠️ No RAG topics selected - all questions will be routed to teams instead of getting AI responses.")
        elif len(selected_topics) == len(all_topics):
            st.info("💡 All topics will use AI responses - no questions will be routed to teams.")

    with tab5:
        st.markdown("### User Interface Settings")

        st.session_state.rag_settings['show_analysis'] = st.checkbox(
            "Show analysis by default in chat",
            value=st.session_state.rag_settings['show_analysis'],
            help="Show classification and search details by default in chat interface"
        )

        st.markdown("### Color Customization")
        st.info("Color customization for topics, priorities, and sentiments coming soon!")

    # Settings validation warnings
    st.markdown("---")
    st.markdown("### ⚠️ Configuration Warnings")

    warnings = []
    settings = st.session_state.rag_settings

    # Check for potentially problematic configurations
    # Note: Weight balance is automatically maintained, so no need to check for both weights being 0

    if settings['enable_hybrid_search'] and settings['hybrid_vector_weight'] == 0:
        warnings.append("⚠️ Vector search disabled (weight = 0) - only keyword search will be used, may miss semantic matches")

    if settings['enable_hybrid_search'] and settings['hybrid_keyword_weight'] == 0:
        warnings.append("💡 Keyword search disabled (weight = 0) - only vector search will be used, may miss exact term matches")

    if settings['temperature'] > 1.5:
        warnings.append("⚠️ High temperature (>1.5) may produce inconsistent responses")

    if settings['top_k'] > 10:
        warnings.append("⚠️ High TOP_K (>10) may include less relevant results and increase token usage")

    if settings['max_tokens'] < 200:
        warnings.append("⚠️ Low max_tokens (<200) may result in truncated responses")

    if settings['score_threshold'] > 0.8:
        warnings.append("⚠️ High score threshold (>0.8) may result in no search results")

    # Check collection-specific warnings
    if settings.get('collection_name'):
        collection_info, info_error = get_collection_info(settings['collection_name'])
        if info_error:
            warnings.append(f"⚠️ Collection '{settings['collection_name']}' may not exist or be accessible")
        elif collection_info and collection_info['points_count'] == 0:
            warnings.append(f"⚠️ Collection '{settings['collection_name']}' is empty - no search results will be returned")

    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.success("✅ Configuration looks good!")

    # Apply settings section
    st.markdown("---")
    st.markdown("### Apply Changes")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if st.button("🔄 Apply Settings", type="primary", help="Apply current settings to RAG pipeline"):
            if st.session_state.rag_pipeline:
                try:
                    # Apply settings to the RAG pipeline
                    success = st.session_state.rag_pipeline.update_settings(st.session_state.rag_settings)
                    if success:
                        st.success("✅ Settings applied successfully!")
                        st.info("💡 Settings are now active for new queries and responses.")
                    else:
                        st.error("❌ Error applying some settings. Check console for details.")
                except Exception as e:
                    st.error(f"❌ Error applying settings: {e}")
            else:
                st.warning("RAG pipeline not initialized")

    with col2:
        if st.button("↩️ Reset to Defaults"):
            st.session_state.rag_settings = {
                'top_k': 5,
                'score_threshold': 0.3,
                'hybrid_vector_weight': 1.0,
                'hybrid_keyword_weight': 0.0,
                'collection_name': 'atlan_docs',
                'max_tokens': 1000,
                'temperature': 0.3,
                'classification_temperature': 0.1,
                'llm_model': 'gpt-4o',
                'enable_query_enhancement': False,
                'enable_hybrid_search': True,
                'show_analysis': True,
                'rag_topics': ['How-to', 'Product', 'Best practices', 'API/SDK', 'SSO']
            }
            st.success("🔄 Settings reset to defaults")
            st.rerun()

    with col3:
        if st.button("📥 Export Settings"):
            settings_json = json.dumps(st.session_state.rag_settings, indent=2)
            st.download_button(
                label="💾 Download JSON",
                data=settings_json,
                file_name="rag_settings.json",
                mime="application/json"
            )

    # Current settings display
    st.markdown("---")
    st.markdown("### Settings Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📝 Configured Settings (in UI)")
        with st.expander("View UI settings", expanded=False):
            st.json(st.session_state.rag_settings)

    with col2:
        st.markdown("#### ⚡ Active Pipeline Settings")
        if st.session_state.rag_pipeline:
            try:
                active_settings = st.session_state.rag_pipeline.get_current_settings()
                with st.expander("View active settings", expanded=False):
                    st.json(active_settings)
            except Exception as e:
                st.error(f"Could not retrieve active settings: {e}")
        else:
            st.warning("Pipeline not initialized")

    # Add import settings functionality
    st.markdown("---")
    st.markdown("### Import Settings")

    uploaded_file = st.file_uploader(
        "📁 Upload settings JSON file",
        type=['json'],
        help="Upload a previously exported settings file"
    )

    if uploaded_file is not None:
        try:
            settings_data = json.loads(uploaded_file.read().decode('utf-8'))

            # Validate settings structure
            required_keys = {'top_k', 'score_threshold', 'temperature', 'llm_model'}
            if required_keys.issubset(settings_data.keys()):
                if st.button("🔄 Import and Apply Settings"):
                    st.session_state.rag_settings.update(settings_data)
                    if st.session_state.rag_pipeline:
                        success = st.session_state.rag_pipeline.update_settings(settings_data)
                        if success:
                            st.success("✅ Settings imported and applied successfully!")
                        else:
                            st.error("❌ Settings imported but failed to apply to pipeline")
                    else:
                        st.success("✅ Settings imported! Apply them when pipeline is ready.")
                    st.rerun()
            else:
                st.error("❌ Invalid settings file format. Missing required keys.")
        except json.JSONDecodeError:
            st.error("❌ Invalid JSON file format")
        except Exception as e:
            st.error(f"❌ Error reading settings file: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; font-size: 14px;">
    Built with Streamlit, OpenAI GPT-4o, FastEmbed, and Qdrant Vector Database
</div>
""", unsafe_allow_html=True)
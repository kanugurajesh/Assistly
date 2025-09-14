# Atlan Customer Support Copilot

An advanced AI-powered customer support system that automatically classifies tickets and provides intelligent responses using state-of-the-art Retrieval-Augmented Generation (RAG) with hybrid search, query enhancement, and optimized chunking strategies.

## 🌟 Features

### Core Functionality
- **Bulk Ticket Classification**: Automatically classify 30+ sample tickets with AI-powered categorization
- **Interactive AI Agent**: Real-time chat interface for new ticket submission and response
- **Conversational Memory**: Context-aware conversations using LangChain ChatMessageHistory with in-memory storage
- **Smart Classification**: Topic tags, sentiment analysis, and priority assignment
- **Advanced RAG Responses**: Intelligent answers powered by hybrid search and enhanced retrieval
- **Source Citations**: All responses include links to relevant documentation
- **Search Transparency**: Real-time indicators showing search methods used (vector, keyword, or hybrid)
- **Dynamic Settings Management**: Comprehensive settings page for real-time pipeline configuration

### Advanced RAG Features
- **Hybrid Search**: Combines vector similarity and BM25 keyword search for optimal relevance
- **Query Enhancement**: GPT-4o powered query expansion for technical terms (configurable)
- **Enhanced Chunking**: Code block preservation with intelligent markdown structure awareness
- **Smart Reranking**: Weighted merging of vector and keyword search results
- **Quality Metrics**: Chunk quality indicators including code detection and header analysis
- **Real-time Configuration**: Dynamic settings updates without application restart
- **Settings Import/Export**: JSON-based configuration backup and sharing

### Classification Schema
- **Topic Tags**: How-to, Product, Connector, Lineage, API/SDK, SSO, Glossary, Best practices, Sensitive data
- **Sentiment**: Frustrated, Curious, Angry, Neutral
- **Priority**: P0 (High), P1 (Medium), P2 (Low)

## 🎯 Major Design Decisions & Trade-offs

### 1. Multi-Stage Pipeline Architecture
**Decision**: Separate data pipeline (scraping → storage → vectorization) from deployment application.

**Why**:
- **Data Persistence**: Web scraping is expensive and rate-limited. MongoDB storage allows reprocessing embeddings without re-scraping.
- **Deployment Flexibility**: App folder contains only deployment dependencies, enabling clean Streamlit Cloud deployment.
- **Development Efficiency**: Can iterate on AI logic without re-running expensive data collection.
- **A/B Testing**: Separate collections enable comparison between basic and enhanced RAG implementations.

**Trade-off**: Increased complexity vs. reliability, cost efficiency, and experimentation capability.

### 2. Advanced Technology Stack Choices

#### Hybrid Search: Vector + BM25 vs. Pure Vector Search
**Decision**: Implement hybrid search combining vector similarity and BM25 keyword search.

**Why**:
- **Technical Term Precision**: BM25 excels at exact matches for technical terms, APIs, and product names.
- **Semantic Understanding**: Vector search captures conceptual relationships and context.
- **Complementary Strengths**: Vector search for "how to authenticate" + BM25 for "SAML SSO" = comprehensive coverage.
- **Fallback Strategy**: Graceful degradation to vector-only if BM25 fails.

**Trade-off**: System complexity and processing overhead vs. significantly improved retrieval quality for technical documentation.

#### Query Enhancement: GPT-4o Expansion vs. Direct Search
**Decision**: Optional GPT-4o query enhancement with configurable toggle.

**Why**:
- **Technical Term Expansion**: "SSO" → "SAML single sign-on authentication setup"
- **Context Enrichment**: "API rate limits" → "REST API rate limiting configuration and best practices"
- **Acronym Resolution**: Critical for technical documentation where acronyms are prevalent.
- **Cost Control**: Configurable feature allows optimization for different use cases.

**Trade-off**: Additional API costs and latency vs. dramatically improved retrieval for technical queries.

#### Enhanced Chunking: Code-Aware vs. Simple Character Splitting
**Decision**: Advanced recursive splitting with code block preservation and quality metrics.

**Why**:
- **Code Integrity**: Preserves ```code blocks``` as single units to maintain functional examples.
- **Structure Awareness**: Respects markdown headers, lists, and procedures.
- **Quality Tracking**: Metadata enables optimization and debugging of retrieval quality.
- **Context Preservation**: Smart boundaries prevent splitting related instructions.

**Trade-off**: Processing complexity and storage overhead vs. significantly better content quality and retrieval accuracy.

### 3. Feature Toggle Architecture
**Decision**: Configurable enhancement toggles rather than fixed implementation.

**Why**:
- **Deployment Flexibility**: Different environments can optimize for cost vs. quality.
- **Performance Tuning**: Disable expensive features for high-volume scenarios.
- **Gradual Rollout**: Test advanced features incrementally in production.
- **User Choice**: Let users balance speed vs. comprehensive results.

**Trade-off**: Configuration complexity vs. deployment flexibility and performance optimization.

### 4. Smart Reranking Strategy
**Decision**: 70/30 weighted fusion of vector and BM25 results with intelligent deduplication.

**Why**:
- **Balanced Relevance**: Vector search weighted higher for semantic understanding.
- **Exact Match Boost**: BM25 results get significant weight for technical precision.
- **Deduplication**: Documents found by both methods receive relevance boost.
- **Empirical Optimization**: 70/30 split tested for optimal balance in technical documentation.

**Trade-off**: Algorithm complexity vs. superior result ranking and relevance.

### 5. Dual Collection Strategy
**Decision**: Separate "enhanced" and "standard" Qdrant collections for A/B testing.

**Why**:
- **Performance Comparison**: Direct measurement of advanced features' impact.
- **Risk Mitigation**: Fallback to standard collection if enhanced features fail.
- **Feature Validation**: Quantitative assessment of enhancement value.
- **Gradual Migration**: Safe transition from basic to advanced implementations.

**Trade-off**: Storage overhead and maintenance complexity vs. risk reduction and optimization capability.

### 6. Technology Stack for Advanced RAG

#### MongoDB + Qdrant vs. Single Database
**Decision**: Dual storage with enhanced Qdrant collections for hybrid search.

**Why**:
- **Data Integrity**: MongoDB preserves original content for reprocessing and debugging.
- **Hybrid Performance**: Qdrant's vector capabilities + in-memory BM25 for keyword search.
- **Collection Management**: Separate enhanced collections for advanced features.
- **Backup Strategy**: Multiple data preservation layers prevent data loss.

**Trade-off**: Infrastructure complexity vs. performance, flexibility, and data safety.

#### OpenAI GPT-4o vs. Local Models
**Decision**: OpenAI GPT-4o for classification, response generation, and query enhancement.

**Why**:
- **Quality**: Superior reasoning for complex ticket classification and technical query expansion.
- **JSON Reliability**: Consistent structured output for automated processing.
- **Context Window**: Large context enables conversation memory and comprehensive responses.
- **Development Speed**: No model training, fine-tuning, or hosting infrastructure needed.

**Trade-off**: Ongoing API costs vs. response quality, development speed, and advanced capabilities.

#### FastEmbed BGE-small + rank-bm25 vs. Single Approach
**Decision**: Hybrid embedding strategy with local FastEmbed and in-memory BM25.

**Why**:
- **Cost Efficiency**: Free local embeddings vs. OpenAI embedding API costs.
- **Privacy**: Document content never leaves local environment.
- **Performance**: 384-dim embeddings balance quality with speed.
- **Hybrid Capability**: BM25 enables exact term matching for technical precision.

**Trade-off**: Implementation complexity vs. cost savings, privacy, and enhanced search capabilities.

## 🏗️ Architecture

### Complete System Architecture with Component Interactions

```
                           🌐 USER INTERFACE LAYER
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                    👤 User Browser Session                                 │
    │  ┌─────────────────────────────────────────────────────────────────────┐   │
    │  │  📊 Dashboard   💬 Chat Agent   ⚙️ Settings   📈 Analytics Page   │   │
    │  │  • Bulk Class.  • Real-time Chat • Dynamic Config • Performance    │   │
    │  │  • 30+ Tickets  • Memory Context  • Import/Export  • Search Stats   │   │
    │  │  • Statistics   • Source Cites    • Validation    • Usage Metrics   │   │
    │  └─────────────────────────────────────────────────────────────────────┘   │
    └─────────────────────────┬───────────────────────────────────────────────────┘
                             │ HTTP Requests
                             ▼
                   🖥️ STREAMLIT APPLICATION LAYER
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                         main.py (Port 8501)                                │
    │  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────────┐   │
    │  │   UI Controls   │   │  Session State  │   │    Event Handlers       │   │
    │  │ • Input Forms   │   │ • User Session  │   │  • Button Clicks        │   │
    │  │ • Display Logic │   │ • Memory Store  │   │  • Text Input           │   │
    │  │ • File Uploads  │   │ • Chat History  │   │  • Page Navigation      │   │
    │  └─────────────────┘   └─────────────────┘   └─────────────────────────┘   │
    └─────────────────────────┬───────────────────────────────────────────────────┘
                              │ Function Calls
                              ▼
                   🧠 AI PROCESSING LAYER (rag_pipeline.py)
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                      Advanced RAG Pipeline Engine                          │
    │  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────────┐   │
    │  │Classification   │   │  Query Pipeline │   │   Response Generator    │   │
    │  │ • Topic Tags    │   │ • Enhancement   │   │ • Template Rendering   │   │
    │  │ • Sentiment     │   │ • Hybrid Search │   │ • Citation Assembly     │   │
    │  │ • Priority      │   │ • Smart Rerank  │   │ • Context Integration   │   │
    │  └─────────────────┘   └─────────────────┘   └─────────────────────────┘   │
    └──────┬──────────────────┬───────────────────────────┬──────────────────────┘
           │                  │                           │
           ▼                  ▼                           ▼
        🤖 EXTERNAL AI APIs                🗄️ DATA STORAGE LAYER
    ┌─────────────────┐     ┌────────────────────────────────────────────────────┐
    │   OpenAI GPT-4o │     │                 Database Services                  │
    │ ┌─────────────┐ │     │  ┌─────────────────┐   ┌─────────────────────────┐ │
    │ │Classification│ │──▶│  │   MongoDB Atlas │   │     Qdrant Cloud        │ │
    │ │• JSON Output │ │    │  │ ┌─────────────┐ │   │ ┌─────────────────────┐ │ │
    │ │• Structured │ │     │  │ │Raw Documents│ │   │ │Vector Collections   │ │ │
    │ └─────────────┘ │     │  │ │• HTML Text  │ │   │ │• atlan_docs_enhanced│ │ │
    │ ┌─────────────┐ │     │  │ │• Metadata   │ │   │ │• Embeddings (384d)  │ │ │
    │ │Query Enhance│ │     │  │ │• Timestamps │ │   │ │• Payloads           │ │ │
    │ │• Term Expand│ │     │  │ └─────────────┘ │   │ └─────────────────────┘ │ │
    │ │• Tech Terms │ │     │  │ ┌─────────────┐ │   │ ┌─────────────────────┐ │ │
    │ └─────────────┘ │     │  │ │Backup Files │ │   │ │In-Memory BM25 Index │ │ │
    │ ┌─────────────┐ │     │  │ │• JSON Dumps │ │   │ │• Keyword Search     │ │ │
    │ │RAG Response │ │     │  │ │• Recovery   │ │   │ │• TF-IDF Scoring     │ │ │
    │ │• Contextual │ │     │  │ └─────────────┘ │   │ │• rank-bm25 Library  │ │ │
    │ │• Cited      │ │     │  └─────────────────┘   │ └─────────────────────┘ │ │
    │ └─────────────┘ │     └──────────┬─────────────────────┬─────────────── ── ┘
    └─────────────────┘                │                    │
           ▲                           │                    │
           │ HTTPS/REST API            │                    │
           │                           ▼                    ▼
                              📝 PIPELINE SCRIPTS LAYER
    ┌───────────────────────────────────────────────────────────────────────────┐
    │                        Data Processing Pipeline                           │
    │  ┌─────────────────┐   ┌────────────────────────────────────────────────┐ │
    │  │   scrape.py     │   │              qdrant_ingestion.py               │ │
    │  │ ┌─────────────┐ │   │ ┌─────────────┐   ┌─────────────┐              │ │
    │  │ │Firecrawl API│ │   │ │Text Chunking│   │FastEmbed BGE│              │ │
    │  │ │• Rate Limits│ │   │ │• 1200 tokens│   │• Local Gen  │              │ │
    │  │ │• Content    │ │   │ │• 200 overlap│   │• 384 dims   │              │ │
    │  │ │  Extraction │ │   │ │• Code Aware │   │• Privacy    │              │ │
    │  │ └─────────────┘ │   │ └─────────────┘   └─────────────┘              │ │
    │  │ ┌─────────────┐ │   │ ┌─────────────┐   ┌─────────────┐              │ │
    │  │ │MongoDB Save │ │   │ │Quality      │   │Qdrant Upload│              │ │
    │  │ │• Documents  │ │   │ │Metrics      │   │• Collections│              │ │
    │  │ │• Metadata   │ │   │ │• Code Detect│   │• Vectors    │              │ │
    │  │ │• Backup     │ │   │ │• Headers    │   │• Payloads   │              │ │
    │  │ └─────────────┘ │   │ └─────────────┘   └─────────────┘              │ │
    │  └─────────────────┘   └────────────────────────────────────────────────┘ │
    └───────────────────────────────────────────────────────────────────────────┘
           ▲                                       ▲
           │ Manual Execution                      │ Manual Execution
           │                                       │
                              🌐 DATA SOURCES
    ┌────────────────────────────────────────────────────────────────────────────┐
    │                           External Documentation                           │
    │  ┌─────────────────────────────────────┐  ┌───────────────────────────────┐│
    │  │         docs.atlan.com              │  │     developer.atlan.com       ││
    │  │ • Product Documentation (~700 pages)│  │ • API Documentation (~300)    ││ 
    │  │ • User Guides                       │  │ • SDK References              ││
    │  │ • Feature Explanations              │  │ • Code Examples               ││
    │  │ • Best Practices                    │  │ • Technical Specifications    ││
    │  └─────────────────────────────────────┘  └───────────────────────────────┘│
    └────────────────────────────────────────────────────────────────────────────┘

            🔄 KEY INTERACTION FLOWS:

            📥 DATA PIPELINE FLOW:
            docs.atlan.com → Firecrawl API → scrape.py → MongoDB → qdrant_ingestion.py → Qdrant

            🔍 REAL-TIME SEARCH FLOW:
            User Query → Query Enhancement (GPT-4o) → Hybrid Search (Vector+BM25) →
            Smart Reranking → Context Assembly → Response Generation (GPT-4o) → User

            💬 CHAT INTERACTION FLOW:
            User Input → Streamlit UI → rag_pipeline.py → Classification (GPT-4o) →
            RAG/Routing Decision → Search & Generate → Display with Citations

            ⚙️ CONFIGURATION FLOW:
            .env Variables → Feature Toggles → Pipeline Behavior → Performance Optimization

            🔒 ERROR HANDLING & RECOVERY:
            • MongoDB Backup Files for Data Recovery
            • Graceful Degradation: Hybrid → Vector-only → Routing
            • Rate Limiting with Exponential Backoff
            • Session State Management for UI Persistence
```

### System Components
- **Data Pipeline** (Root scripts): Web scraping → Storage → Vector preparation
- **Deployment** (App folder): Streamlit application with AI capabilities
- **AI Services**: OpenAI for classification and response generation
- **Storage**: MongoDB for documents, Qdrant for vector search

## 🛠️ Tech Stack

### AI/ML
- **OpenAI GPT-4o**: LLM for classification, response generation, and query enhancement
- **FastEmbed BAAI/bge-small-en-v1.5**: Vector embeddings for semantic search (384 dimensions)
- **Qdrant Cloud**: Vector database with hybrid search capabilities
- **rank-bm25**: BM25 algorithm for keyword search and hybrid retrieval
- **LangChain**: Enhanced text processing with advanced chunking strategies

### Application
- **Streamlit**: Interactive web application framework
- **Python**: Core application logic and AI pipeline
- **MongoDB**: Document storage for scraped content

### UI/UX
- **Streamlit Components**: Dashboard and chat interface
- **Custom CSS**: Styled components and responsive design
- **Interactive Elements**: Real-time classification and response generation

### Data Sources & Pipeline
- **Firecrawl API**: Automated web scraping service for documentation
- **docs.atlan.com**: Product documentation and user guides
- **developer.atlan.com**: API and SDK documentation
- **MongoDB**: Persistent storage for all scraped content with metadata
- **Qdrant**: Vector database for semantic search and RAG retrieval

### Pipeline Stages
1. **Web Scraping**: Firecrawl crawls documentation sites and extracts content
2. **Document Storage**: Raw content stored in MongoDB with full metadata
3. **Vector Processing**: Content chunked and embedded using FastEmbed BGE-small
4. **RAG Deployment**: Streamlit app queries Qdrant for relevant context

## 📋 Prerequisites

### For Deployment (Streamlit App)
- Python 3.8+ and pip
- OpenAI API key
- Qdrant Cloud instance (vector database)
- MongoDB Atlas instance (document storage)
- Firecrawl API key (if running custom scraping)

### Project Structure
- **Root directory**: Data pipeline scripts (scrape.py, qdrant_ingestion.py)
- **app/ directory**: Streamlit deployment application with own requirements and .env

## 🚀 Quick Start

### 1. Clone and Setup Environment

```bash
git clone https://github.com/kanugurajesh/Assistly
cd Assistly
```

**Project Structure Overview:**
```
crawling/
├── app/                    # Streamlit deployment
│   ├── main.py            # Main Streamlit application
│   ├── rag_pipeline.py    # AI pipeline implementation
│   ├── requirements.txt   # App dependencies
│   ├── .env.example       # Environment template
│   └── sample_tickets.json
├── memory_manager.py      # Conversational memory management
├── scrape.py              # Firecrawl web scraping
├── qdrant_ingestion.py    # Vector database ingestion
├── requirements.txt       # Data pipeline dependencies
└── README.md
```

Create `.env` file in the `app/` directory (copy from `app/.env.example`):
```env
OPENAI_API_KEY=your_openai_api_key
QDRANT_URI=your_qdrant_cloud_endpoint
QDRANT_API_KEY=your_qdrant_api_key
MONGODB_URI=your_mongodb_atlas_connection_string
FIRECRAWL_API_KEY=your_firecrawl_api_key
```

### 2. Install Dependencies

**For deployment (Streamlit app):**
```bash
pip install -r app/requirements.txt
```

**For data pipeline (if running scraping/ingestion):**
```bash
pip install -r requirements.txt
```

### 3. Data Pipeline Setup (Optional - for custom data)

**Step 1: Web Scraping with Firecrawl**
```bash
# Basic scraping (pre-completed for Atlan docs)
python scrape.py https://docs.atlan.com --limit 700
python scrape.py https://developer.atlan.com --limit 300

# Custom scraping examples
python scrape.py https://your-docs.com --limit 500 --collection custom_docs
```
*All scraped content automatically stored in MongoDB with metadata and backup files.*

**Step 2: Enhanced Vector Database Ingestion**
```bash
# Create enhanced collection with advanced chunking
python qdrant_ingestion.py --qdrant-collection "atlan_docs_enhanced" --recreate

# Advanced ingestion with source filtering
python qdrant_ingestion.py --source-url "https://docs.atlan.com" --qdrant-collection "atlan_docs_enhanced"

# Incremental updates (recommended for production)
python qdrant_ingestion.py --qdrant-collection "atlan_docs_enhanced"
```
*Enhanced chunking preserves code blocks, creates quality metrics, and generates embeddings with FastEmbed BGE-small for hybrid search.*

**Note**: The application comes with pre-processed data, so this step is only needed for custom datasets or updates. For advanced configuration options, see the "Advanced Pipeline Options" section below.

## 🔧 Advanced Pipeline Options

### Scraping Configuration (scrape.py)

**Basic Command Structure:**
```bash
python scrape.py <URL> [OPTIONS]
```

**Available Options:**
- `--limit <number>`: Maximum pages to crawl (default: 700)
- `--collection <name>`: MongoDB collection name (default: atlan_developer_docs)

**Common Scraping Scenarios:**
```bash
# Scrape with custom page limit
python scrape.py https://docs.atlan.com --limit 500

# Scrape to custom MongoDB collection
python scrape.py https://docs.atlan.com --collection custom_docs

# Scrape developer docs with different limits
python scrape.py https://developer.atlan.com --limit 200 --collection dev_docs
```

### Ingestion Configuration (qdrant_ingestion.py)

**Advanced Command Structure:**
```bash
python qdrant_ingestion.py [OPTIONS]
```

**Available Options:**
- `--source-url <url>`: Filter documents by specific source URL
- `--collection <name>`: MongoDB collection name (default: atlan_developer_docs)
- `--qdrant-collection <name>`: Qdrant collection name (default: atlan_docs)
- `--recreate`: Delete and recreate Qdrant collection (removes existing data)
- `--no-incremental`: Process all documents (skip duplicate checking)

**Advanced Ingestion Examples:**
```bash
# Process only developer documentation
python qdrant_ingestion.py --source-url "https://developer.atlan.com"

# Process only general documentation
python qdrant_ingestion.py --source-url "https://docs.atlan.com"

# Recreate collection (fresh start)
python qdrant_ingestion.py --recreate

# Process all documents without incremental checking
python qdrant_ingestion.py --no-incremental

# Process custom collection with filtering
python qdrant_ingestion.py --collection custom_docs --source-url "https://example.com"

# Create custom Qdrant collection
python qdrant_ingestion.py --qdrant-collection "developer_vectors"

# Process custom MongoDB collection to custom Qdrant collection
python qdrant_ingestion.py --collection dev_docs --qdrant-collection "dev_vectors"

# Full rebuild with specific source and custom collection
python qdrant_ingestion.py --recreate --source-url "https://developer.atlan.com" --qdrant-collection "dev_only"
```

## 📂 Document Filtering & Collection Management

### Source URL Filtering Benefits

**Selective Processing:**
- Update only specific documentation domains
- Test pipeline with subset of data
- Separate processing schedules for different sites

**Document Type Classification:**
- Automatic categorization: `developer.atlan.com` → "developer" type
- All other sources → "docs" type
- Enables filtered search and analytics

**Performance Optimization:**
- Process only changed documentation
- Reduce vector database update time
- Minimize embedding generation costs

**Custom Qdrant Collections:**
- Separate vector collections for different projects
- Independent collection lifecycle management
- Isolated testing and production environments
- Multiple documentation versions in parallel

### Collection Management Workflows

**Development & Testing:**
```bash
# Create test collection with limited data
python scrape.py https://docs.atlan.com --limit 50 --collection test_docs
python qdrant_ingestion.py --collection test_docs --qdrant-collection test_vectors --recreate
```

**Production Updates:**
```bash
# Incremental update (default behavior)
python qdrant_ingestion.py --source-url "https://docs.atlan.com"

# Full rebuild when needed
python qdrant_ingestion.py --recreate
```

**Multi-Source Management:**
```bash
# Separate ingestion for different documentation types
python qdrant_ingestion.py --source-url "https://developer.atlan.com"
python qdrant_ingestion.py --source-url "https://docs.atlan.com"
```

### Incremental Processing

**How It Works:**
- Checks MongoDB document IDs already in Qdrant
- Skips processing of existing documents
- Only processes new or updated content

**When to Use `--no-incremental`:**
- After modifying chunking parameters
- When reprocessing is needed due to embedding model changes
- For debugging or validation purposes

### 4. Run the Application

**Run the Streamlit app:**
```bash
cd app
streamlit run main.py
```

The application will open automatically in your browser at `http://localhost:8501`

### 5. Streamlit Deployment

**Deploy to Streamlit Community Cloud:**
1. Push your repository to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set main file path: `app/main.py`
5. Add environment variables in Streamlit Cloud settings
6. Deploy your application

## 📖 Usage Guide

### Dashboard Page
1. Navigate to "📊 Dashboard" in the sidebar
2. Click "Load & Classify All Tickets"
3. View AI-generated classifications for all 30+ sample tickets
4. Analyze summary statistics and topic distributions
5. Search and examine individual ticket classifications

### Interactive Agent Page
1. Navigate to "💬 Chat Agent" in the sidebar
2. Enter your question in the chat interface
3. Toggle "Show internal analysis" to view classification details
4. Get intelligent responses with source citations
5. Experience context-aware conversations with memory
6. Use the "Conversation Management" sidebar to view memory stats or clear history
7. Try sample questions or submit your own tickets

## 💬 Conversational Memory Features

### Context-Aware Conversations
The system maintains conversation history to provide context-aware responses:
- **Follow-up Questions**: Ask related questions without repeating context
- **Reference Previous Answers**: The AI remembers what it told you earlier
- **Natural Flow**: Conversations feel more natural and coherent

### Memory Management
**Conversation Management Sidebar:**
- **Memory Statistics**: View active sessions and total message count
- **Current Session Info**: See number of exchanges in current conversation
- **Clear History**: Manually reset conversation memory when needed

**Automatic Features:**
- **Session Isolation**: Each browser session has its own conversation memory
- **Message Limits**: Automatically trims to last 20 messages to prevent token overflow
- **Auto Expiry**: Sessions expire after 60 minutes of inactivity
- **Smart Trimming**: Removes oldest messages while preserving conversation pairs

### Example Conversation Flow
```
User: "How do I connect Snowflake to Atlan?"
AI: "To connect Snowflake to Atlan, you need to configure..." [provides detailed steps]

User: "What permissions do I need for this?"
AI: "For the Snowflake connection we discussed, you'll need..." [remembers previous context]

User: "Are there any security considerations?"
AI: "Yes, for your Snowflake-Atlan integration, consider..." [builds on conversation]
```

### Technical Implementation
- **Backend**: LangChain's `InMemoryChatMessageHistory` for pure RAM storage
- **No Database**: Conversations stored in Python dictionaries (no external dependencies)
- **Session Management**: UUID-based session identification with Streamlit session state
- **Context Integration**: Previous conversation included in RAG prompts for better responses

### Settings Page
1. Navigate to "⚙️ Settings" in the sidebar
2. **Collection Management**: Select from available Qdrant collections with real-time discovery
3. **Collection Information**: View collection points, vector size, and distance metrics
4. Configure search parameters (TOP_K, score thresholds, hybrid weights)
5. Adjust model settings (temperature, max tokens, model selection)
6. Toggle features (hybrid search, query enhancement)
7. Customize UI preferences (show analysis default)
8. Apply settings in real-time without restarting the application
9. Export/import settings configurations as JSON files
10. View configuration warnings for potentially problematic settings
11. **Troubleshooting**: Built-in connection diagnostics and collection validation

## 🧠 Advanced AI Pipeline Details

### Enhanced Classification Logic
The system analyzes tickets using structured prompts to generate:
1. **Topic Tags**: Multiple relevant categories with high accuracy
2. **Sentiment**: Emotional tone analysis for prioritization
3. **Priority**: Business impact assessment with context awareness

### Advanced RAG Response Logic
- **RAG Topics**: How-to, Product, Best practices, API/SDK, SSO → Generate answers using hybrid search
- **Routing Topics**: Connector, Lineage, Glossary, Sensitive data → Route to specialized teams
- **Query Processing**: Optional GPT-4o enhancement expands technical terms
- **Search Strategy**: Hybrid vector + keyword search with smart reranking
- **Response Generation**: Context-aware answers with source attribution

### Advanced RAG Pipeline Components

#### 1. Query Enhancement Pipeline (Optional)
- **Input**: Raw user query (e.g., "How to setup SSO?")
- **Processing**: GPT-4o expands technical terms and acronyms
- **Output**: Enhanced query (e.g., "How to configure SAML single sign-on authentication in Atlan?")
- **Benefits**: Better retrieval for technical documentation
- **Toggle**: Configurable via `ENABLE_QUERY_ENHANCEMENT`

#### 2. Hybrid Search System
- **Vector Search**: Semantic similarity using FastEmbed BGE-small (384 dim)
- **Keyword Search**: BM25 algorithm for exact term matching
- **Fusion Strategy**: 70/30 weighted combination with smart deduplication
- **Reranking**: Boosts documents found by both methods
- **Fallback**: Graceful degradation to vector-only if BM25 fails

#### 3. Enhanced Chunking Strategy
- **Structure Preservation**: Special handling for code blocks and headers
- **Smart Separators**: 15+ separator types for optimal boundaries
- **Quality Metrics**: Tracks code presence, headers, and word count
- **Metadata Enhancement**: Chunk-level quality indicators
- **Context Maintenance**: Preserves related content together

### Enhanced Chunking Strategy
- **Chunk Size**: 1200 tokens with 200 token overlap
- **Method**: Advanced recursive character splitting with enhanced separators
- **Code Preservation**: Special handling for ``` code blocks and indented code
- **Structure Awareness**: Preserves headers, lists, procedures, and markdown formatting
- **Quality Metrics**: Tracks code blocks, headers, word count, and chunk quality scores
- **Smart Boundaries**: 15+ separator types for optimal semantic chunking

### Hybrid Search System
- **Vector Search**: BAAI/bge-small-en-v1.5 (384 dimensions) with cosine similarity
- **Keyword Search**: BM25 algorithm for exact term matching
- **Search Fusion**: 70/30 weighted combination of vector and keyword results
- **Smart Reranking**: Deduplication and relevance scoring with boost for multi-method matches
- **Score Threshold**: 0.3 minimum similarity for vector results
- **Top-K Retrieval**: 5 most relevant chunks from hybrid results

### Conversational Memory System
- **Memory Backend**: LangChain's `InMemoryChatMessageHistory` for pure RAM storage
- **Session Management**: Unique session IDs for each browser session with automatic timeout
- **Context Window**: Last 5 message exchanges included in RAG prompts for continuity
- **Memory Features**:
  - Automatic message trimming (max 20 messages per session)
  - Session cleanup and expiration (60-minute timeout)
  - Manual conversation clearing via UI
  - Memory usage statistics and monitoring
- **No External Dependencies**: Pure in-memory storage without databases

## 🔧 Configuration Options

### Environment Variables (app/.env)
- `OPENAI_API_KEY`: Required for GPT-4o classification, response generation, and query enhancement
- `QDRANT_URI`: Qdrant Cloud vector database endpoint for hybrid search
- `QDRANT_API_KEY`: Authentication for Qdrant Cloud instance
- `MONGODB_URI`: MongoDB Atlas connection string for document storage
- `FIRECRAWL_API_KEY`: Firecrawl API key for web scraping (data pipeline only)

### Advanced RAG Configuration (app/rag_pipeline.py)
- `ENABLE_QUERY_ENHANCEMENT`: Toggle GPT-4o query expansion (default: False)
- `ENABLE_HYBRID_SEARCH`: Toggle vector + BM25 hybrid search (default: True)
- `HYBRID_VECTOR_WEIGHT`: Weight for vector search results (default: 1.0)
- `HYBRID_KEYWORD_WEIGHT`: Weight for BM25 keyword results (default: 0.0)
- `COLLECTION_NAME`: Qdrant collection name (default: "atlan_docs_enhanced")
- `SCORE_THRESHOLD`: Minimum similarity threshold (default: 0.3)
- `TOP_K`: Number of search results to retrieve (default: 5)
- `MAX_TOKENS`: Maximum response length (default: 1000)
- `TEMPERATURE`: Response creativity level (default: 0.3)
- `LLM_MODEL`: OpenAI model for responses (default: "gpt-4o")

### Dynamic Settings Management
- **Real-time Updates**: All configuration changes apply immediately without restart
- **Collection Management**: Dynamic Qdrant collection discovery and switching
- **Settings Validation**: Built-in warnings for potentially problematic configurations
- **Import/Export**: JSON-based settings backup and sharing capabilities
- **UI Integration**: Settings page with tabbed interface for different parameter categories
- **Configuration Persistence**: Settings stored in session state and applied to pipeline
- **Connection Diagnostics**: Real-time collection validation and troubleshooting
- **Fallback Handling**: Graceful degradation when settings cause issues

### Data Pipeline Configuration
- **Scraping Parameters**: Use `--limit` and `--collection` options in scrape.py for custom URLs and crawl limits
- **Source Filtering**: Use `--source-url` in qdrant_ingestion.py for selective document processing
- **Collection Management**: Use `--qdrant-collection`, `--recreate` and `--no-incremental` options for collection lifecycle
- **Custom Collections**: Use `--qdrant-collection` to create separate vector collections for different projects
- **Chunk Configuration**: Adjust size and overlap in qdrant_ingestion.py (default: 1200 tokens, 200 overlap)
- **Vector Search**: Modify threshold and top-K in app/rag_pipeline.py (default: 0.3 threshold, 5 chunks)

### Common Pipeline Workflows

**Complete Fresh Setup:**
```bash
# Scrape new documentation
python scrape.py https://new-docs.com --limit 500 --collection new_docs

# Create fresh vector database
python qdrant_ingestion.py --collection new_docs --recreate
```

**Incremental Updates (Recommended):**
```bash
# Re-scrape updated content (overwrites existing URLs)
python scrape.py https://docs.atlan.com --limit 700

# Incremental ingestion (only new/changed documents)
python qdrant_ingestion.py
```

**Domain-Specific Processing:**
```bash
# Update only developer documentation vectors
python qdrant_ingestion.py --source-url "https://developer.atlan.com"

# Update only general documentation vectors
python qdrant_ingestion.py --source-url "https://docs.atlan.com"
```

**Testing and Development:**
```bash
# Create test dataset
python scrape.py https://docs.atlan.com --limit 20 --collection test_data

# Test ingestion pipeline with custom Qdrant collection
python qdrant_ingestion.py --collection test_data --qdrant-collection test_vectors --recreate
```

**Multiple Project Management:**
```bash
# Project A: Customer documentation
python scrape.py https://customer-docs.com --collection customer_docs
python qdrant_ingestion.py --collection customer_docs --qdrant-collection customer_vectors

# Project B: Internal documentation
python scrape.py https://internal-docs.com --collection internal_docs
python qdrant_ingestion.py --collection internal_docs --qdrant-collection internal_vectors
```

### Application Customization
- **Classification Prompts**: Edit prompts in app/rag_pipeline.py for custom categorization
- **Response Templates**: Modify RAG and routing responses in app/main.py
- **UI Styling**: Update custom CSS in app/main.py for branding

## 📊 Performance Metrics & Advanced Features

### Enhanced Data Pipeline Efficiency
- **Smart Scraping**: Firecrawl with automated content extraction and metadata preservation
- **Persistent Storage**: MongoDB with backup capabilities and incremental processing
- **Advanced Vector Ingestion**: Batch processing with enhanced chunking and quality metrics
- **Hybrid Search Performance**: Combined vector + keyword search with intelligent reranking

### Advanced Response Quality Measures
- **Multi-Method Retrieval**: Hybrid search combines semantic and keyword matching
- **Query Enhancement**: GPT-4o expands technical terms for better retrieval (configurable)
- **Smart Reranking**: 70/30 weighted fusion of vector and BM25 results
- **Source Attribution**: All RAG responses include original documentation URLs
- **Relevance Scoring**: Vector similarity + BM25 scoring with threshold 0.3
- **Context Quality**: Top-5 chunks from hybrid results for comprehensive answers
- **Search Transparency**: Real-time indicators showing search methods used

### Advanced Scalability Features
- **Feature Toggles**: Configurable query enhancement and hybrid search
- **Collection Management**: Separate enhanced and standard collections
- **Incremental Processing**: Skip already processed documents for efficiency
- **Quality Metrics**: Chunk-level quality indicators (code detection, headers, word count)
- **Error Resilience**: Graceful fallbacks for all advanced features
- **Performance Monitoring**: Search method tracking and optimization insights

## 🚨 Troubleshooting

### Data Pipeline Issues

**1. Firecrawl Scraping Problems**
- Verify Firecrawl API key in environment variables
- Check rate limits and adjust scraping delays in scrape.py
- Monitor MongoDB connection for storage issues

**2. MongoDB Storage Issues**
- Validate MongoDB Atlas connection string
- Check database and collection permissions
- Verify network access to MongoDB cluster

**3. Vector Database Problems**
- Verify Qdrant Cloud instance is accessible
- Check embedding dimensions match (384 for BGE-small)
- Validate collection exists and has correct configuration

### Application Issues

**4. Streamlit Deployment**
- Ensure all environment variables are set in app/.env
- Check that app/requirements.txt includes all dependencies
- Verify OpenAI API key has sufficient credits

**5. Classification Errors**
- Review prompt templates in app/rag_pipeline.py
- Check JSON parsing logic for malformed responses
- Monitor OpenAI API rate limits

### Debugging Tips
- Check Streamlit logs for detailed error messages
- Validate environment variable loading in app directory
- Test individual pipeline components (MongoDB, Qdrant, OpenAI)
- Monitor API usage and rate limits across all services

## 📝 Development Notes

### Advanced Project Structure Philosophy
- **Separation of Concerns**: Clean separation between data pipeline (root) and deployment (app/)
- **Environment Isolation**: Each tier has independent requirements and configuration
- **Feature Modularity**: Advanced RAG features can be toggled independently
- **Data Persistence**: MongoDB enables reprocessing and experimentation
- **Deployment Ready**: Enhanced app/ folder with advanced search capabilities

### Enhanced Architecture Decisions
1. **Advanced Data Pipeline**: Firecrawl → MongoDB → Enhanced Qdrant → Advanced Streamlit
2. **Hybrid Search System**: Vector + BM25 keyword search with intelligent fusion
3. **Query Enhancement**: Optional GPT-4o query expansion for technical terms
4. **Enhanced Chunking**: Code-aware splitting with quality metrics
5. **Smart Configuration**: Feature toggles for different deployment scenarios
6. **Dual Collection Strategy**: Standard vs enhanced collections for comparison
7. **Performance Optimization**: Configurable search weights and thresholds

### Advanced Trade-offs & Design Decisions
- **Query Enhancement**: Optional GPT-4o expansion vs direct search (configurable)
- **Hybrid Search**: Vector + keyword complexity vs pure vector simplicity
- **Enhanced Chunking**: Structure preservation vs simple character splitting
- **Feature Toggles**: Flexibility vs configuration complexity
- **Dual Collections**: Comparison capability vs storage overhead
- **Search Transparency**: User insight vs UI complexity
- **Performance vs Features**: Configurable enhancement levels for different use cases

### Production-Ready Enhancements
- **Collection Management**: Enhanced vs standard collections for A/B testing
- **Feature Flags**: Runtime configuration of advanced features
- **Quality Metrics**: Chunk-level quality indicators for optimization
- **Search Analytics**: Real-time method tracking and performance insights
- **Graceful Degradation**: Fallbacks ensure system reliability

### Enhanced RAG Implementation (advanced-rag-enhancements branch)
| Feature | Implementation |
|---------|----------------|
| **Search Method** | Hybrid vector + BM25 keyword search |
| **Query Processing** | Optional GPT-4o query enhancement |
| **Chunking** | Code-aware splitting with quality metrics |
| **Results** | Smart reranking with configurable fusion weights |
| **UI Feedback** | Search method indicators + transparency |
| **Collection** | `atlan_docs_enhanced` with enhanced metadata |
| **Configurability** | Feature toggles for all enhancements |
| **Performance** | Graceful degradation and fallbacks |
| **Settings Management** | Dynamic configuration with real-time updates |
| **Configuration** | Import/export, validation, and persistence |

### Key Improvements
1. **✅ Better Technical Term Handling**: Hybrid search excels at exact matches
2. **✅ Enhanced Code Examples**: Preserved code blocks in chunking
3. **✅ Query Expansion**: GPT-4o expands acronyms and technical terms
4. **✅ Search Transparency**: Users see which methods found their answers
5. **✅ Quality Metrics**: Chunk-level indicators for optimization
6. **✅ Configurable Features**: Toggle enhancements based on needs
7. **✅ Dynamic Settings Management**: Real-time configuration without restart
8. **✅ Settings Import/Export**: JSON-based configuration sharing and backup
9. **✅ Collection Management**: Real-time Qdrant collection discovery and switching
10. **✅ Connection Diagnostics**: Built-in troubleshooting for collection issues

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Update documentation
5. Submit pull request

## 📄 License

[MIT License](LICENSE)

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review API documentation
3. Create an issue with detailed reproduction steps


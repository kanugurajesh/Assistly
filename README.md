# Atlan Customer Support Copilot

An AI-powered customer support system that automatically classifies tickets and provides intelligent responses using Retrieval-Augmented Generation (RAG) with Atlan's documentation.

## 🌟 Features

### Core Functionality
- **Bulk Ticket Classification**: Automatically classify 30+ sample tickets with AI-powered categorization
- **Interactive AI Agent**: Real-time chat interface for new ticket submission and response
- **Smart Classification**: Topic tags, sentiment analysis, and priority assignment
- **RAG Responses**: Intelligent answers powered by Atlan's documentation
- **Source Citations**: All responses include links to relevant documentation

### Classification Schema
- **Topic Tags**: How-to, Product, Connector, Lineage, API/SDK, SSO, Glossary, Best practices, Sensitive data
- **Sentiment**: Frustrated, Curious, Angry, Neutral
- **Priority**: P0 (High), P1 (Medium), P2 (Low)

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Next.js       │    │   Python         │    │   Databases     │
│   Frontend      │    │   AI Pipeline    │    │                 │
├─────────────────┤    ├──────────────────┤    ├─────────────────┤
│ • Dashboard     │◄───┤ • RAG Pipeline   │◄───┤ • MongoDB       │
│ • Chat UI       │    │ • Classification │    │   (Raw docs)    │
│ • API Routes    │    │ • Embeddings     │    │ • Qdrant        │
└─────────────────┘    └──────────────────┘    │   (Vectors)     │
                                               └─────────────────┘
           │                        │                     ▲
           ▼                        ▼                     │
    ┌─────────────┐         ┌─────────────┐              │
    │ Gemini API  │         │ Firecrawl   │──────────────┘
    │ • LLM       │         │ • Web       │
    │ • Embeddings│         │   Scraping  │
    └─────────────┘         └─────────────┘
```

## 🛠️ Tech Stack

### AI/ML
- **Google Gemini 1.5 Flash**: LLM for classification and response generation
- **Gemini text-embedding-001**: Vector embeddings for semantic search
- **Qdrant**: Vector database for RAG retrieval
- **LangChain**: Text processing and chunking

### Backend
- **Next.js 15**: Full-stack React framework with API routes
- **TypeScript**: Type-safe development
- **MongoDB**: Document storage for scraped content

### Frontend
- **React 19**: Modern UI components
- **Tailwind CSS**: Utility-first styling
- **Responsive Design**: Mobile-friendly interface

### Data Sources
- **docs.atlan.com**: Product documentation
- **developer.atlan.com**: API and SDK documentation
- **Firecrawl**: Web scraping service

## 📋 Prerequisites

- Node.js 18+ and npm
- Python 3.8+ and pip
- Google AI API key
- Qdrant Cloud instance
- MongoDB Atlas instance
- Firecrawl API key

## 🚀 Quick Start

### 1. Clone and Setup Environment

```bash
git clone <repository-url>
cd crawling
```

Create `.env` file:
```env
GOOGLE_API_KEY=your_gemini_api_key
QDRANT_URI=your_qdrant_endpoint
QDRANT_API_KEY=your_qdrant_api_key
MONGODB_URI=your_mongodb_connection_string
FIRECRAWL_API_KEY=your_firecrawl_api_key
```

### 2. Install Dependencies

**Python dependencies:**
```bash
pip install -r requirements.txt
```

**Node.js dependencies:**
```bash
cd app
npm install
```

### 3. Data Pipeline Setup

**Step 1: Scrape Documentation**
```bash
# Scrape developer.atlan.com (already done)
python scrape.py

# Optional: Scrape docs.atlan.com
# Update scrape.py URL and run again
```

**Step 2: Ingest to Vector Database**
```bash
# Process MongoDB documents and create embeddings
python qdrant-ingestion.py
```

**Step 3: Process Sample Tickets (Optional)**
```bash
# Classify all sample tickets for testing
python process_tickets.py
```

### 4. Run the Application

**Development mode:**
```bash
cd app
npm run dev
```

**Production build:**
```bash
cd app
npm run build
npm start
```

Access the application at `http://localhost:3000`

## 📖 Usage Guide

### Dashboard (/dashboard)
1. Click "Load & Classify All Tickets"
2. View AI-generated classifications for all 30 sample tickets
3. Analyze summary statistics and topic distributions
4. Examine individual ticket classifications

### Interactive Agent (/chat)
1. Enter your question in the chat interface
2. View the internal AI analysis (classification details)
3. Get intelligent responses with source citations
4. Try sample questions or submit your own tickets

### API Endpoints

- `GET /api/tickets/bulk-classify` - Classify all sample tickets
- `POST /api/tickets/classify` - Classify individual ticket
- `POST /api/chat` - Interactive agent with RAG
- `GET/POST /api/knowledge/search` - Search documentation

## 🧠 AI Pipeline Details

### Classification Logic
The system analyzes tickets using structured prompts to generate:
1. **Topic Tags**: Multiple relevant categories
2. **Sentiment**: Emotional tone analysis
3. **Priority**: Business impact assessment

### RAG Response Logic
- **RAG Topics**: How-to, Product, Best practices, API/SDK, SSO → Generate answers using documentation
- **Routing Topics**: Connector, Lineage, Glossary, Sensitive data → Route to appropriate teams

### Chunking Strategy
- **Chunk Size**: 1200 tokens with 200 token overlap
- **Method**: Recursive character splitting with markdown awareness
- **Preservation**: Code blocks, tables, and lists as single units

### Vector Search
- **Embedding Model**: text-embedding-001 (768 dimensions)
- **Search Strategy**: Cosine similarity with score threshold 0.3
- **Top-K Retrieval**: 5 most relevant chunks

## 🔧 Configuration Options

### Environment Variables
- `GOOGLE_API_KEY`: Required for AI services
- `QDRANT_URI`: Vector database endpoint
- `QDRANT_API_KEY`: Authentication for Qdrant
- `MONGODB_URI`: Document storage connection
- `FIRECRAWL_API_KEY`: Web scraping service

### Customizable Parameters
- Chunk size and overlap (`qdrant-ingestion.py`)
- Vector search threshold and top-K
- Classification prompt engineering
- Response generation templates

## 📊 Performance Metrics

### Response Quality Measures
- **Source Attribution**: All RAG responses include documentation URLs
- **Relevance Scoring**: Vector similarity scores for retrieved chunks
- **Classification Consistency**: Structured output with validation

### Scalability Features
- **Batch Processing**: Efficient embedding generation
- **Rate Limiting**: API-friendly request handling
- **Error Handling**: Graceful fallbacks and retries

## 🚨 Troubleshooting

### Common Issues

**1. API Rate Limits**
- Implement exponential backoff
- Reduce batch sizes in `qdrant-ingestion.py`

**2. Vector Search Issues**
- Verify Qdrant collection exists
- Check embedding dimensions match (768)
- Validate API credentials

**3. Classification Errors**
- Review prompt templates in `rag_pipeline.py`
- Check JSON parsing logic
- Verify Gemini API access

### Debugging Tips
- Enable verbose logging in Python scripts
- Use browser dev tools for frontend issues
- Check Next.js API route responses
- Validate environment variable loading

## 🎯 Future Enhancements

### Short-term Improvements
- Add conversation history and context
- Implement user feedback collection
- Enhanced error handling and validation
- Performance optimization for large datasets

### Long-term Roadmap
- Multi-language support
- Advanced analytics dashboard
- Integration with ticketing systems
- Custom model fine-tuning
- Real-time collaboration features

## 📝 Development Notes

### Architecture Decisions
1. **Hybrid Approach**: Python for AI pipeline, TypeScript for web application
2. **Vector Database**: Qdrant chosen for performance and cloud availability
3. **Embedding Strategy**: Task-specific embeddings (retrieval_document vs retrieval_query)
4. **Response Generation**: Balance between speed (Flash) and quality

### Trade-offs
- **Latency vs Quality**: Gemini Flash for speed over Pro for accuracy
- **Storage vs Compute**: Pre-computed embeddings vs real-time generation
- **Complexity vs Maintainability**: Modular architecture with clear separation

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

# Atlan Customer Support Copilot - Streamlit Version

This is the Streamlit implementation of the Atlan Customer Support Copilot, providing the same functionality as the Next.js version with a simpler deployment approach.

## Features

### 🏠 Home Page
- Feature overview and navigation
- Key features showcase
- Easy access to Dashboard and Chat Agent

### 📊 Dashboard
- Bulk ticket classification of 30+ sample tickets
- Real-time AI-powered categorization
- Summary statistics with visual indicators
- Searchable ticket table
- Topic, sentiment, and priority classification

### 💬 Interactive AI Agent
- Real-time chat interface
- Dual view: Internal analysis + Final response
- RAG-powered responses for supported topics
- Automatic routing for specialized issues
- Source citations from Atlan documentation
- Sample questions for easy testing

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**
   Create a `.env` file in the app directory with:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   QDRANT_URI=your_qdrant_endpoint
   QDRANT_API_KEY=your_qdrant_api_key
   MONGODB_URI=your_mongodb_connection_string
   FIRECRAWL_API_KEY=your_firecrawl_api_key
   ```

3. **Run Data Pipeline** (from parent directory)
   ```bash
   python qdrant_ingestion.py
   ```

4. **Start Streamlit App**
   ```bash
   streamlit run main.py
   ```

5. **Access the Application**
   Open your browser to `http://localhost:8501`

## Architecture

The Streamlit app integrates with the existing Python backend:

- **RAG Pipeline**: Uses `rag_pipeline.py` for AI processing
- **Sample Data**: Loads from `sample_tickets.json`
- **Vector Database**: Connects to Qdrant for document retrieval
- **AI Models**: OpenAI GPT-4o for classification and responses

## Navigation

- Use the sidebar to navigate between pages
- **Home**: Overview and quick access buttons
- **Dashboard**: Bulk ticket processing and analysis
- **Chat Agent**: Interactive Q&A interface
- **Settings**: Comprehensive configuration management

## ⚙️ Settings Page Features

### Collection Management
- **Dynamic Discovery**: Automatically detects available Qdrant collections
- **Real-time Information**: Shows collection stats (points count, vector dimensions, distance metric)
- **Easy Switching**: Dropdown selection with immediate application
- **Validation**: Built-in checks for collection existence and compatibility
- **Troubleshooting**: Connection diagnostics and helpful error messages

### Search Configuration
- **TOP_K**: Configure number of search results to retrieve (1-20)
- **Score Threshold**: Set minimum similarity threshold for relevance (0.0-1.0)
- **Hybrid Weights**: Balance vector search vs keyword search (0.0-1.0 each)

### Model Settings
- **OpenAI Model Selection**: Choose between GPT-4o, GPT-4o-mini, GPT-4-turbo
- **Response Temperature**: Control creativity vs consistency (0.0-2.0)
- **Max Tokens**: Set maximum response length (100-4000)
- **Classification Temperature**: Fine-tune classification consistency (0.0-1.0)

### Feature Toggles
- **Hybrid Search**: Enable/disable vector + keyword search combination
- **Query Enhancement**: Toggle GPT-4o query expansion for technical terms

### Settings Management
- **Real-time Updates**: All changes apply immediately without restart
- **Import/Export**: JSON-based configuration backup and sharing
- **Reset to Defaults**: One-click restoration of default settings
- **Validation Warnings**: Automatic detection of problematic configurations
- **Settings Overview**: View both UI settings and active pipeline configuration

## Key Differences from Next.js Version

1. **Integrated Application**: All functionality with settings management
2. **Server-Side Processing**: No API routes needed
3. **Direct Integration**: Calls Python functions directly
4. **Advanced Configuration**: Dynamic settings page with real-time updates
5. **Collection Management**: Real-time Qdrant collection discovery and switching
6. **Simplified Deployment**: Single command to run with comprehensive features

## Deployment Options

### Local Development
```bash
streamlit run main.py
```

### Streamlit Community Cloud
1. Push to GitHub repository
2. Connect repository to Streamlit Community Cloud
3. Set environment variables in dashboard
4. Deploy automatically

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Performance Considerations

- **Caching**: Streamlit's built-in caching for RAG pipeline initialization
- **Session State**: Maintains conversation history and classified tickets
- **Progressive Loading**: Shows progress bars for long-running operations
- **Error Handling**: Graceful fallbacks for API failures

## Customization

### Styling
- Modify CSS in the `st.markdown()` sections
- Adjust colors in the color helper functions
- Update layout using Streamlit columns

### Functionality
- Add new sample questions in the chat section
- Modify classification display logic
- Extend analysis views

## Troubleshooting

1. **RAG Pipeline Initialization Failed**
   - Check environment variables
   - Ensure Qdrant database is populated
   - Verify API keys are valid

2. **Sample Tickets Not Loading**
   - Ensure `sample_tickets.json` exists in parent directory
   - Check file permissions and format

3. **Classification Errors**
   - Verify OpenAI API key
   - Check rate limits and quota
   - Monitor API usage costs

## Support

For issues specific to the Streamlit version, check:
1. Streamlit documentation: https://docs.streamlit.io
2. Environment variable configuration
3. Python package versions in requirements.txt
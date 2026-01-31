# Streamlit Chat Interface

Professional chat UI for the DataChat-RAG healthcare AdTech Q&A system.

## Features

### Chat Interface
- Clean, professional message display
- User/assistant message differentiation
- Animated message appearances
- Streaming response support

### Sidebar Controls
- **Connection Status**: Real-time API connectivity indicator
- **Theme Toggle**: Light/dark mode
- **Conversation Management**: New chat, clear history
- **Query Type Filter**: Filter responses by type (All/SQL/Docs/Hybrid)
- **Session Stats**: Questions asked, query type breakdown
- **Sample Questions**: Quick-start questions to try
- **System Health**: Component status monitoring

### Message Features
- **Query Type Badge**: Visual indicator (SQL/DOC/HYBRID)
- **Confidence Score**: Bar chart showing classification confidence
- **Processing Time**: Time taken to generate response
- **Expandable SQL Query**: Copy button for generated SQL
- **Document Sources**: Expandable source cards with metadata
- **Query Routing Reasoning**: See why queries were classified a certain way
- **Follow-up Suggestions**: Context-aware next questions
- **CSV Download**: Export SQL query results

### Visual Design
- Custom CSS for professional appearance
- Smooth animations and transitions
- Status indicators (connected/disconnected/loading)
- Responsive layout

## Running the Application

### Prerequisites
```bash
pip install streamlit requests pandas
```

### Start the API
```bash
# In one terminal
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Start the Streamlit UI
```bash
# In another terminal
streamlit run src/ui/chat_app.py
```

Or use the entry point:
```bash
streamlit run src/ui/app:main
```

### Access the Application
Open your browser to: `http://localhost:8501`

## Sample Questions

The sidebar includes these sample questions:

1. "What was our total ad spend last month?"
2. "Which campaigns have the highest CTR?"
3. "What are our HIPAA compliance requirements for healthcare ads?"
4. "Compare performance of pharma vs medical device campaigns"
5. "What's the process for getting an ad creative approved?"
6. "Why is campaign X underperforming compared to benchmarks?"
7. "Top 5 campaigns by conversions this quarter"
8. "What are our targeting policies for pharmaceutical ads?"

## Screenshots

### Light Theme
- Clean, professional interface
- Blue accent colors
- Clear message differentiation

### Dark Theme
- Easy on the eyes for extended use
- High contrast for readability
- Same functionality as light theme

## Configuration

### API URL
Set the `API_BASE_URL` environment variable (default: `http://localhost:8000`):
```bash
export API_BASE_URL=http://localhost:8000
```

### API Timeout
Adjust timeout in seconds (default: 120):
```python
API_TIMEOUT = 120
```

## Component Status

The sidebar shows real-time health status for:
- RAG Chain
- Document Retriever (with chunk count)
- SQL Retriever
- Query Router
- Conversation Store

## Keyboard Shortcuts

- `Enter` in chat input: Submit question
- Click sample questions in sidebar: Quick fill

## File Structure

```
src/ui/
├── chat_app.py       # Main Streamlit application
├── __init__.py       # Package exports
└── README.md         # This file
```

## Troubleshooting

### API Not Connected
- Ensure the API is running on the correct port
- Check the API_BASE_URL environment variable
- Verify the API health endpoint is accessible

### No Sample Questions Showing
- Check the sidebar is expanded
- Scroll down to see the "Try These" section

### Download CSV Not Available
- Only appears for SQL query results
- Check if the query returned data

### Theme Not Persisting
- Theme selection resets on page refresh
- This is expected behavior (per-session)

## Future Enhancements

- [ ] Multi-language support
- [ ] Export entire conversation as PDF
- [ ] Voice input/output
- [ ] Real-time collaboration
- [ ] Advanced analytics dashboard
- [ ] Custom branding options

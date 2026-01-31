/**
 * FraudDocs-RAG Frontend
 *
 * A React chat interface for querying financial fraud detection documents.
 * Features:
 * - Chat interface with question/input
 * - Answer display with source citations
 * - Document type filtering
 * - Loading states
 * - Error handling
 * - Responsive design with Tailwind CSS
 */

import React, { useState, useRef, useEffect } from "react";

// ============================================================================
// Constants
// ============================================================================

const API_BASE_URL = "http://localhost:8000";

const DOCUMENT_TYPES = [
  { value: "", label: "All Documents" },
  { value: "aml", label: "AML (Anti-Money Laundering)" },
  { value: "kyc", label: "KYC (Know Your Customer)" },
  { value: "fraud", label: "Fraud Detection" },
  { value: "regulation", label: "Regulations" },
  { value: "general", label: "General" },
];

const CATEGORY_COLORS = {
  aml: "bg-red-100 text-red-800 border-red-200",
  kyc: "bg-blue-100 text-blue-800 border-blue-200",
  fraud: "bg-purple-100 text-purple-800 border-purple-200",
  regulation: "bg-green-100 text-green-800 border-green-200",
  general: "bg-gray-100 text-gray-800 border-gray-200",
};

const CATEGORY_ICONS = {
  aml: "🛡️",
  kyc: "👤",
  fraud: "🔍",
  regulation: "📋",
  general: "📄",
};

// ============================================================================
// Components
// ============================================================================

/**
 * Source citation component with expandable preview
 */
const SourceCitation = ({ source, index }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
      {/* Header */}
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2 flex-1">
          {/* Citation number */}
          <span className="flex-shrink-0 w-6 h-6 bg-indigo-600 text-white rounded-full flex items-center justify-center text-sm font-semibold">
            {index}
          </span>

          {/* Source file name */}
          <span className="font-medium text-gray-900 truncate">
            {source.source}
          </span>
        </div>

        {/* Category badge */}
        <span
          className={`px-2 py-1 rounded-full text-xs font-medium border ${
            CATEGORY_COLORS[source.doc_type] || CATEGORY_COLORS.general
          }`}
        >
          {CATEGORY_ICONS[source.doc_type] || "📄"} {source.doc_type.toUpperCase()}
        </span>
      </div>

      {/* Score */}
      <div className="flex items-center gap-1 mb-2">
        <span className="text-xs text-gray-500">Relevance:</span>
        <div className="flex-1 bg-gray-200 rounded-full h-2">
          <div
            className="bg-indigo-600 h-2 rounded-full transition-all"
            style={{ width: `${source.score * 100}%` }}
          />
        </div>
        <span className="text-xs text-gray-600 font-medium">
          {(source.score * 100).toFixed(0)}%
        </span>
      </div>

      {/* Expandable preview */}
      <div className="mt-3">
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-sm text-indigo-600 hover:text-indigo-800 font-medium flex items-center gap-1"
        >
          {isExpanded ? "▼" : "▶"} {isExpanded ? "Hide" : "Show"} preview
        </button>

        {isExpanded && (
          <div className="mt-2 p-3 bg-gray-50 rounded border border-gray-200">
            <p className="text-sm text-gray-700 whitespace-pre-wrap">
              {source.preview}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * Chat message component
 */
const ChatMessage = ({ message }) => {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div
        className={`max-w-3xl ${
          isUser ? "order-2" : "order-1"
        }`}
      >
        {/* Message header */}
        <div className="flex items-center gap-2 mb-2">
          <div
            className={`w-8 h-8 rounded-full flex items-center justify-center text-sm ${
              isUser
                ? "bg-indigo-600 text-white"
                : "bg-gray-300 text-gray-700"
            }`}
          >
            {isUser ? "👤" : "🤖"}
          </div>
          <span className="text-sm font-medium text-gray-700">
            {isUser ? "You" : "FraudDocs-RAG"}
          </span>
          {message.timestamp && (
            <span className="text-xs text-gray-500">
              {new Date(message.timestamp).toLocaleTimeString()}
            </span>
          )}
        </div>

        {/* Message content */}
        <div
          className={`${
            isUser
              ? "bg-indigo-600 text-white"
              : "bg-white border border-gray-200 text-gray-900"
          } rounded-2xl px-4 py-3 ${
            isUser ? "rounded-tr-sm" : "rounded-tl-sm"
          }`}
        >
          {message.role === "assistant" ? (
            <div className="prose prose-sm max-w-none">
              {/* Parse and render answer with citations */}
              <div
                dangerouslySetInnerHTML={{
                  __html: message.content.answer.replace(
                    /\[(\d+)\]/g,
                    '<span class="inline-flex items-center justify-center w-5 h-5 bg-indigo-100 text-indigo-800 rounded-full text-xs font-semibold mx-0.5">[$1]</span>'
                  ),
                }}
              />
            </div>
          ) : (
            <p className="text-sm">{message.content}</p>
          )}
        </div>

        {/* Sources for assistant messages */}
        {message.role === "assistant" && message.content.sources && (
          <div className="mt-3 space-y-2">
            <div className="text-sm font-medium text-gray-700">
              📚 Sources ({message.content.sources.length})
            </div>
            <div className="space-y-2">
              {message.content.sources.map((source, idx) => (
                <SourceCitation
                  key={idx}
                  source={source}
                  index={idx + 1}
                />
              ))}
            </div>
          </div>
        )}

        {/* Processing time */}
        {message.processingTime && (
          <div className="mt-2 text-xs text-gray-500">
            ⏱️ Processed in {message.processingTime.toFixed(2)}s
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * Loading animation component
 */
const LoadingSpinner = () => (
  <div className="flex items-center gap-2 text-gray-600">
    <div className="animate-spin rounded-full h-6 w-6 border-2 border-indigo-600 border-t-transparent" />
    <span>Analyzing documents and generating response...</span>
  </div>
);

/**
 * Error message component
 */
const ErrorMessage = ({ message, onDismiss }) => (
  <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
    <span className="text-red-600 text-xl">⚠️</span>
    <div className="flex-1">
      <h4 className="font-medium text-red-800">Error</h4>
      <p className="text-sm text-red-700 mt-1">{message}</p>
    </div>
    <button
      onClick={onDismiss}
      className="text-red-600 hover:text-red-800 text-xl leading-none"
    >
      ×
    </button>
  </div>
);

/**
 * Welcome message component
 */
const WelcomeMessage = () => (
  <div className="bg-gradient-to-br from-indigo-50 to-blue-50 border border-indigo-100 rounded-xl p-6 text-center">
    <div className="text-4xl mb-4">🏦</div>
    <h2 className="text-2xl font-bold text-gray-900 mb-2">
      Welcome to FraudDocs-RAG
    </h2>
    <p className="text-gray-700 mb-4">
      Your intelligent assistant for financial fraud detection and compliance documents.
    </p>
    <div className="text-sm text-gray-600 space-y-1">
      <p>Ask questions about:</p>
      <div className="flex flex-wrap justify-center gap-2 mt-2">
        {["AML", "KYC", "Fraud Detection", "Regulations"].map((topic) => (
          <span
            key={topic}
            className="px-3 py-1 bg-white rounded-full text-xs font-medium border border-indigo-200"
          >
            {topic}
          </span>
        ))}
      </div>
    </div>
    <div className="mt-4 text-xs text-gray-500">
      Example questions:
    </div>
    <div className="mt-2 space-y-1 text-sm">
      {[
        "What are the SAR filing deadlines?",
        "Explain Customer Due Diligence requirements",
        "What are the red flags for money laundering?",
      ].map((q, i) => (
        <div key={i} className="text-indigo-700 italic">
          "{q}"
        </div>
      ))}
    </div>
  </div>
);

// ============================================================================
// Main App Component
// ============================================================================

function App() {
  // State
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [docTypeFilter, setDocTypeFilter] = useState("");
  const [healthStatus, setHealthStatus] = useState(null);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Check API health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        setHealthStatus(data);
      } catch (err) {
        console.error("Health check failed:", err);
        setHealthStatus({ status: "unreachable" });
      }
    };

    checkHealth();
    // Check every 30 seconds
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!input.trim() || isLoading) return;

    const question = input.trim();
    setInput("");
    setError(null);

    // Add user message
    const userMessage = {
      role: "user",
      content: question,
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMessage]);

    // Set loading
    setIsLoading(true);

    try {
      // Query the API
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: question,
          doc_type_filter: docTypeFilter || null,
          use_rerank: true,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();

      // Add assistant message
      const assistantMessage = {
        role: "assistant",
        content: {
          answer: data.answer,
          sources: data.sources,
        },
        processingTime: data.processing_time,
        timestamp: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      console.error("Query failed:", err);
      setError(
        err.message || "Failed to process your question. Please try again."
      );
    } finally {
      setIsLoading(false);
    }
  };

  // Handle example question click
  const handleExampleClick = (question) => {
    setInput(question);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="text-3xl">🏦</div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  FraudDocs-RAG
                </h1>
                <p className="text-sm text-gray-600">
                  Financial Compliance & Fraud Detection Assistant
                </p>
              </div>
            </div>

            {/* Health status indicator */}
            {healthStatus && (
              <div className="flex items-center gap-2">
                <span
                  className={`w-2 h-2 rounded-full ${
                    healthStatus.status === "healthy"
                      ? "bg-green-500 animate-pulse"
                      : healthStatus.status === "unreachable"
                      ? "bg-red-500"
                      : "bg-yellow-500"
                  }`}
                />
                <span className="text-xs text-gray-600">
                  {healthStatus.status === "healthy"
                    ? "Connected"
                    : healthStatus.status === "unreachable"
                    ? "API Unreachable"
                    : "Degraded"}
                </span>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-6xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar */}
          <aside className="lg:col-span-1">
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-4 sticky top-6">
              <h3 className="font-semibold text-gray-900 mb-4">
                🔍 Search Filters
              </h3>

              {/* Document type filter */}
              <div className="mb-4">
                <label
                  htmlFor="docTypeFilter"
                  className="block text-sm font-medium text-gray-700 mb-2"
                >
                  Document Type
                </label>
                <select
                  id="docTypeFilter"
                  value={docTypeFilter}
                  onChange={(e) => setDocTypeFilter(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 text-sm"
                >
                  {DOCUMENT_TYPES.map((type) => (
                    <option key={type.value} value={type.value}>
                      {type.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* Info box */}
              <div className="bg-indigo-50 rounded-lg p-3 text-sm">
                <p className="font-medium text-indigo-900 mb-1">
                  💡 Tip
                </p>
                <p className="text-indigo-700 text-xs">
                  Filter by document type to get more targeted results from
                  specific regulatory areas.
                </p>
              </div>

              {/* Stats */}
              {healthStatus?.collection_stats && (
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <h4 className="text-sm font-medium text-gray-700 mb-2">
                    📊 Statistics
                  </h4>
                  <div className="text-xs text-gray-600 space-y-1">
                    <div>
                      Documents:{" "}
                      <span className="font-semibold">
                        {healthStatus.collection_stats.total_docs || "N/A"}
                      </span>
                    </div>
                    <div>
                      Environment:{" "}
                      <span className="font-semibold capitalize">
                        {healthStatus.environment}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </aside>

          {/* Chat area */}
          <div className="lg:col-span-3">
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 flex flex-col" style={{ minHeight: "600px" }}>
              {/* Messages area */}
              <div className="flex-1 overflow-y-auto p-6">
                {messages.length === 0 ? (
                  <WelcomeMessage />
                ) : (
                  <div>
                    {messages.map((msg, idx) => (
                      <ChatMessage key={idx} message={msg} />
                    ))}
                    {isLoading && (
                      <div className="flex justify-start mb-4">
                        <div className="bg-white border border-gray-200 rounded-2xl rounded-tl-sm px-4 py-3">
                          <LoadingSpinner />
                        </div>
                      </div>
                    )}
                    <div ref={messagesEndRef} />
                  </div>
                )}
              </div>

              {/* Error display */}
              {error && (
                <div className="px-6 pb-4">
                  <ErrorMessage
                    message={error}
                    onDismiss={() => setError(null)}
                  />
                </div>
              )}

              {/* Input area */}
              <div className="border-t border-gray-200 p-4 bg-gray-50 rounded-b-xl">
                <form onSubmit={handleSubmit} className="flex gap-3">
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask about AML, KYC, fraud detection, or regulations..."
                    disabled={isLoading}
                    className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
                  />
                  <button
                    type="submit"
                    disabled={isLoading || !input.trim()}
                    className="px-6 py-3 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
                  >
                    {isLoading ? "⏳" : "Send"}
                  </button>
                </form>
                <div className="mt-2 text-xs text-gray-500 flex items-center justify-between">
                  <span>
                    Press Enter to send • Shift+Enter for new line
                  </span>
                  {messages.length > 0 && (
                    <button
                      onClick={() => setMessages([])}
                      className="text-red-600 hover:text-red-700 font-medium"
                    >
                      Clear Chat
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-8 py-4 border-t border-gray-200 bg-white">
        <div className="max-w-6xl mx-auto px-4 text-center text-sm text-gray-600">
          <p>
            FraudDocs-RAG v1.0.0 | Built with LlamaIndex, ChromaDB, and FastAPI
          </p>
          <p className="mt-1 text-xs text-gray-500">
            ⚠️ This is an AI assistant. Always verify with official regulatory
            sources for compliance decisions.
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;

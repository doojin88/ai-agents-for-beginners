# Semantic Kernel Azure Search RAG - Complete Explanation

## 📋 Overview

This notebook demonstrates a **Retrieval-Augmented Generation (RAG)** implementation using Semantic Kernel with Azure AI Search. The example creates an AI travel agent that retrieves documents from an Azure search index, integrates supplementary data sources, and streams responses with full transparency of function calls.

### Key Capabilities
- **Document Retrieval**: Search and retrieve travel documents from Azure AI Search
- **Multi-Plugin Architecture**: Separate plugins for document search and weather information
- **Function Calling**: Automatic LLM-driven function invocation based on user queries
- **Streaming Responses**: Real-time response delivery with function call transparency
- **Conversation Memory**: Thread-based context management across multiple turns

---

## 🏗️ Architecture Components

### 1. Azure AI Search (Vector Database)

**Purpose**: Persistent storage and retrieval of travel documents

```python
# Index Configuration
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchableField(name="content", type=SearchFieldDataType.String)
]

index = SearchIndex(name="travel-documents", fields=fields)
```

**Characteristics**:
- **Index Name**: `travel-documents`
- **Schema**: Simple (ID + searchable content field)
- **Persistence**: Idempotent initialization - checks if index exists before creating
- **Search Type**: Keyword search (not semantic/vector search)
- **Sample Data**: 5 documents about Contoso Travel services

**Sample Documents**:
1. Luxury vacation packages to exotic destinations
2. Premium travel services with personalized planning
3. Travel insurance coverage details
4. Popular destinations (Maldives, Swiss Alps, African safaris)
5. Exclusive access to boutique hotels and guided tours

### 2. Semantic Kernel Agent

**Purpose**: Orchestrate conversation flow and function execution

```python
agent = ChatCompletionAgent(
    service=chat_completion_service,          # OpenAI service
    plugins=[SearchPlugin(search_client), WeatherInfoPlugin()],  # Available tools
    name="TravelAgent",                       # Agent identifier
    instructions="Answer travel queries using the provided tools and context..."
)
```

**Configuration**:
- **Service**: AsyncOpenAI client (GitHub Models inference endpoint)
- **Model**: `gpt-4o-mini` (cost-effective, streaming-capable)
- **Plugins**: Registered as tools available to the LLM
- **Instructions**: System-level guidance to prevent hallucination

### 3. LLM Service Configuration

```python
load_dotenv()
client = AsyncOpenAI(
    api_key=os.environ["GITHUB_TOKEN"],
    base_url="https://models.inference.ai.azure.com/"
)

chat_completion_service = OpenAIChatCompletion(
    ai_model_id="gpt-4o-mini",
    async_client=client,
)
```

**Features**:
- Asynchronous client for non-blocking I/O
- GitHub Models integration for cost-effective inference
- Streaming support enabled by default

---

## 🔌 Plugin Architecture

### SearchPlugin (RAG Core)

**Responsibility**: Document retrieval and context augmentation

#### Function 1: `retrieve_documents`

```python
@kernel_function(
    name="retrieve_documents",
    description="Retrieve documents from the Azure Search service."
)
def get_retrieval_context(self, query: str) -> str:
    results = self.search_client.search(query)
    context_strings = [
        f"Document: {result['content']}"
        for result in results
    ]
    return "\\n\\n".join(context_strings) if context_strings else "No results found"
```

**How it works**:
1. Accepts user query as string
2. Executes keyword search against Azure Search index
3. Formats results as "Document: {content}" strings
4. Joins multiple results with newlines
5. Returns formatted context or "No results found"

**LLM Decision Point**: The LLM automatically calls this function when the user asks a question that might benefit from document retrieval.

#### Function 2: `build_augmented_prompt`

```python
@kernel_function(
    name="build_augmented_prompt",
    description="Build an augmented prompt using retrieval context or function results."
)
def build_augmented_prompt(self, query: str, retrieval_context: str) -> str:
    return (
        f"Retrieved Context:\\n{retrieval_context}\\n\\n"
        f"User Query: {query}\\n\\n"
        "First review the retrieved context, if this does not answer the query, "
        "try calling an available plugin functions that might give you an answer. "
        "If no context is available, say so."
    )
```

**Purpose**:
- Structures the augmentation of user queries with retrieval context
- Provides instructions for fallback behavior
- Guides LLM to try other functions if initial context insufficient

**Note**: Defined but not directly invoked in the demonstrated flow

### WeatherInfoPlugin (Supplementary Data)

**Responsibility**: Provide temperature information for travel destinations

```python
class WeatherInfoPlugin:
    def __init__(self):
        self.destination_temperatures = {
            "maldives": "82°F (28°C)",
            "swiss alps": "45°F (7°C)",
            "african safaris": "75°F (24°C)"
        }

    @kernel_function(
        description="Get the average temperature for a specific travel destination."
    )
    def get_destination_temperature(self, destination: str) -> Annotated[str, "Returns the average temperature for the destination."]:
        normalized_destination = destination.lower()

        if normalized_destination in self.destination_temperatures:
            return f"The average temperature in {destination} is {self.destination_temperatures[normalized_destination]}."
        else:
            return f"Sorry, I don't have temperature information for {destination}. Available destinations are: Maldives, Swiss Alps, and African safaris."
```

**Features**:
- Case-insensitive destination matching
- Graceful fallback with available options
- Hardcoded data (could be replaced with real API call)
- Return type annotation for documentation

---

## 🔄 Execution Flow

### Overall Query Processing

```
User Input
    ↓
[Agent receives query via invoke_stream()]
    ↓
[LLM analyzes query and function descriptions]
    ↓
[LLM decides which functions to invoke]
    ↓
[Execute function(s) and gather results]
    ↓
[Stream response chunks to user]
    ├── FunctionCallContent (function name + arguments)
    ├── FunctionResultContent (function output)
    └── StreamingTextContent (LLM-generated text)
    ↓
[Update conversation thread for continuity]
```

### Key Processing Patterns

#### 1. Thread-Based Conversation Management

```python
thread: ChatHistoryAgentThread | None = None

async for response in agent.invoke_stream(
    messages=user_input,
    thread=thread,
):
    thread = response.thread  # Maintain conversation history
    # Process response items
```

**Benefits**:
- Preserves context across multiple turns
- Enables follow-up questions
- Single thread per conversation session

#### 2. Streaming Content Handling

The `invoke_stream()` method yields responses containing multiple content types:

```python
async for response in agent.invoke_stream(messages=user_input, thread=thread):
    content_items = list(response.items)

    for item in content_items:
        if isinstance(item, FunctionCallContent):
            # Handle function invocation
        elif isinstance(item, FunctionResultContent):
            # Handle function result
        elif isinstance(item, StreamingTextContent):
            # Handle LLM text chunk
```

---

## 📡 Streaming Implementation Details

### Three-Phase Response Processing

#### Phase 1: Function Call Buffering

```python
current_function_name = None
argument_buffer = ""

if isinstance(item, FunctionCallContent):
    if item.function_name:
        current_function_name = item.function_name

    # Arguments stream in chunks - accumulate them
    if isinstance(item.arguments, str):
        argument_buffer += item.arguments
```

**Why buffering?**
- Function arguments stream as JSON fragments
- Must accumulate complete JSON before parsing
- Enables robust argument reconstruction

#### Phase 2: Function Result Processing

```python
elif isinstance(item, FunctionResultContent):
    # Finalize pending function call
    if current_function_name:
        formatted_args = argument_buffer.strip()
        try:
            parsed_args = json.loads(formatted_args)
            formatted_args = json.dumps(parsed_args)  # Pretty print
        except Exception:
            pass  # Fallback to raw string

        function_calls.append(
            f"Calling function: {current_function_name}({formatted_args})"
        )
        current_function_name = None
        argument_buffer = ""

    function_calls.append(f"\\nFunction Result:\\n\\n{item.result}")
```

**Error Handling**:
- Graceful fallback if JSON parsing fails
- Maintains function call transparency even with malformed arguments

#### Phase 3: Text Accumulation

```python
elif isinstance(item, StreamingTextContent) and item.text:
    full_response.append(item.text)
```

**Result**: `full_response` is a list of text chunks that form the complete LLM response

### UI Rendering

```python
html_output = (
    "<div style='margin-bottom:10px'>"
    "<details>"
    "<summary style='cursor:pointer; font-weight:bold; color:#0066cc;'>"
    "Function Calls (click to expand)</summary>"
    "<div style='margin:10px; padding:10px; background-color:#f8f8f8; "
    "border:1px solid #ddd; border-radius:4px; white-space:pre-wrap;'>"
    f"{chr(10).join(function_calls)}"
    "</div></details></div>"
)
```

**Features**:
- Collapsible function call details
- Preserves formatting with `white-space:pre-wrap`
- Separates "behind-the-scenes" from user-facing response

---

## 💡 Example Interactions

### Query 1: Document Retrieval

**User**: "Can you explain Contoso's travel insurance coverage?"

**Execution**:
```
1. LLM sees query about "insurance coverage"
2. LLM calls: retrieve_documents(query="Contoso travel insurance coverage")
3. Azure Search returns 4 relevant documents:
   - "Contoso's travel insurance covers medical emergencies, trip cancellations, and lost baggage."
   - "Contoso Travel offers luxury vacation packages to exotic destinations worldwide."
   - "Contoso Travel provides exclusive access to boutique hotels and private guided tours."
   - "Our premium travel services include personalized itinerary planning and 24/7 concierge support."
4. LLM synthesizes response:
   - Medical emergencies coverage
   - Trip cancellation protection
   - Lost baggage compensation
```

**Agent Response**:
```
Contoso's travel insurance coverage includes the following:

1. **Medical Emergencies**: Coverage for unforeseen medical issues while traveling.
2. **Trip Cancellations**: Protection in case you need to cancel your trip.
3. **Lost Baggage**: Compensation for baggage that is lost during your trip.

If you need more specific details, contact Contoso directly or refer to official documentation.
```

### Query 2: Supplementary Data

**User**: "What is the average temperature of the Maldives?"

**Execution**:
```
1. LLM sees query about "temperature"
2. LLM calls: get_destination_temperature(destination="Maldives")
3. WeatherInfoPlugin returns: "The average temperature in Maldives is 82°F (28°C)."
4. LLM responds with temperature information
```

### Query 3: Multi-Function Orchestration

**User**: "What is a good cold destination offered by Contoso and what is its average temperature?"

**Execution**:
```
1. LLM recognizes need for both retrieval AND temperature info
2. First call: retrieve_documents(query="cold destination Swiss Alps")
   → Returns: "Popular destinations include the Maldives, Swiss Alps, and African safaris."
3. Second call: get_destination_temperature(destination="Swiss Alps")
   → Returns: "The average temperature in Swiss Alps is 45°F (7°C)."
4. LLM combines both results in response:
   - Swiss Alps is a Contoso destination
   - Average temperature is 45°F (7°C)
   - Suitable for cold weather travel
```

---

## ✅ Best Practices Demonstrated

### 1. Idempotent Initialization

```python
try:
    existing_index = index_client.get_index(index_name)
    print(f"Index '{index_name}' already exists, using the existing index.")
except Exception:
    print(f"Creating new index '{index_name}'...")
    index_client.create_index(index)
```

**Benefit**: Safe to run multiple times without errors

### 2. Defensive Error Handling

```python
try:
    parsed_args = json.loads(formatted_args)
    formatted_args = json.dumps(parsed_args)
except Exception:
    pass  # Leave as raw string
```

**Benefit**: Graceful degradation if JSON parsing fails

### 3. Clear Function Descriptions

```python
@kernel_function(
    name="retrieve_documents",
    description="Retrieve documents from the Azure Search service."
)
```

**Benefit**: Descriptive names guide LLM to make correct function calls

### 4. Modular Plugin Design

- **SearchPlugin**: Encapsulates all retrieval logic
- **WeatherInfoPlugin**: Separate concern for weather data
- **Agent**: Orchestrates without coupling to implementation

**Benefit**: Easy to add, remove, or modify plugins

### 5. Streaming UX

- Real-time response delivery
- Function call transparency with collapsible details
- Better perceived performance

### 6. Thread-Based Conversation

- Single thread object maintains history
- Enables multi-turn conversations
- Preserves context automatically

---

## ⚠️ Limitations and Enhancement Opportunities

### Current Limitations

| Limitation | Impact | Solution |
|-----------|--------|----------|
| **Keyword Search Only** | Misses semantically similar but lexically different content | Implement vector embeddings with semantic search |
| **Hardcoded Weather Data** | Limited to 3 destinations, manual updates required | Integrate real weather API (OpenWeather, Weather API) |
| **No Re-ranking** | Search results not scored by relevance | Add BM25 or semantic similarity scoring |
| **Limited Error Handling** | Search failures could crash the flow | Add try/catch and fallback strategies |
| **`build_augmented_prompt` Unused** | Function defined but not invoked | Integrate into the main retrieval flow |
| **Simple Schema** | Only ID + content fields | Add metadata fields (date, author, source, etc.) |

### Enhancement Opportunities

#### 1. Implement Vector Search
```python
# Current: Keyword search
results = self.search_client.search(query)

# Enhanced: Semantic search with embeddings
vector = embeddings_service.embed(query)
results = self.search_client.search(
    query=None,
    vector=vector,
    k=5,
    vectors_query_kind="similarity"
)
```

#### 2. Add Real Weather Integration
```python
import aiohttp

async def get_destination_temperature(self, destination: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://api.openweathermap.org/data/2.5/weather?q={destination}"
        ) as resp:
            data = await resp.json()
            return f"Temperature in {destination}: {data['main']['temp']}°C"
```

#### 3. Enhance Index Schema
```python
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchableField(name="content", type=SearchFieldDataType.String),
    SimpleField(name="source", type=SearchFieldDataType.String),
    SimpleField(name="date", type=SearchFieldDataType.DateTimeOffset),
    SimpleField(name="category", type=SearchFieldDataType.String),
    SimpleField(name="relevance_score", type=SearchFieldDataType.Double),
]
```

#### 4. Add Result Re-ranking
```python
def get_retrieval_context(self, query: str) -> str:
    results = self.search_client.search(query)

    # Re-rank by relevance score
    ranked = sorted(
        results,
        key=lambda x: self._calculate_relevance(query, x['content']),
        reverse=True
    )[:5]

    return format_results(ranked)
```

#### 5. Integrate `build_augmented_prompt`
```python
async for response in agent.invoke_stream(messages=user_input, thread=thread):
    # After retrieval, use build_augmented_prompt
    retrieval_context = search_plugin.get_retrieval_context(user_input)
    augmented_prompt = search_plugin.build_augmented_prompt(
        user_input,
        retrieval_context
    )
```

---

## 🎯 Key Takeaways

1. **RAG Architecture**: Combines document retrieval with LLM generation for grounded responses
2. **Function Calling**: Semantic Kernel automatically exposes plugin functions as LLM-callable tools
3. **Multi-Plugin Design**: Separate plugins for different concerns (search, weather, etc.)
4. **Streaming Responses**: Real-time delivery with full transparency of function invocations
5. **Conversation Memory**: Thread-based context preserves multi-turn interactions
6. **Modular & Extensible**: Easy to add new plugins or replace data sources

---

## 📚 Production Readiness

This notebook demonstrates several **production-ready patterns**:

✅ Idempotent resource initialization
✅ Graceful error handling
✅ Streaming for better UX
✅ Clear separation of concerns
✅ Transparent debugging capabilities
✅ Multi-turn conversation support

**Not quite production-ready without**:
- Vector embeddings for semantic search
- Real external API integrations
- Comprehensive error handling
- Result caching for performance
- Rate limiting and circuit breakers
- Comprehensive logging and monitoring

---

## 🔗 Related Concepts

- **Retrieval-Augmented Generation (RAG)**: Combining retrieval with generation for factual accuracy
- **Function Calling**: LLM-driven function invocation based on function descriptions
- **Semantic Kernel**: Microsoft's framework for building LLM-based agents
- **Azure AI Search**: Managed search service for document indexing and retrieval
- **Streaming Responses**: Real-time token delivery for better UX
- **Agent Architecture**: Autonomous systems that can invoke tools and maintain context

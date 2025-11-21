# Semantic Kernel Azure Search RAG - 완전 설명

## 📋 개요

이 노트북은 Semantic Kernel과 Azure AI Search를 사용한 **검색 증강 생성(RAG, Retrieval-Augmented Generation)** 구현을 보여줍니다. 예제는 Azure 검색 인덱스에서 문서를 검색하고, 보조 데이터 소스를 통합하며, 함수 호출의 전체 투명성과 함께 응답을 스트리밍하는 AI 여행 에이전트를 만듭니다.

### 주요 기능
- **문서 검색**: Azure AI Search에서 여행 문서 검색 및 검색
- **다중 플러그인 아키텍처**: 문서 검색과 날씨 정보를 위한 별도의 플러그인
- **함수 호출**: 사용자 쿼리에 따른 자동 LLM 기반 함수 호출
- **스트리밍 응답**: 함수 호출 투명성과 함께 실시간 응답 전달
- **대화 메모리**: 여러 턴에 걸친 스레드 기반 컨텍스트 관리

---

## 🏗️ 아키텍처 구성 요소

### 1. Azure AI Search (벡터 데이터베이스)

**목적**: 여행 문서의 지속적 저장 및 검색

```python
# 인덱스 구성
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchableField(name="content", type=SearchFieldDataType.String)
]

index = SearchIndex(name="travel-documents", fields=fields)
```

**특성**:
- **인덱스 이름**: `travel-documents`
- **스키마**: 간단함 (ID + 검색 가능한 콘텐츠 필드)
- **지속성**: 멱등성 초기화 - 생성 전에 인덱스 존재 여부 확인
- **검색 유형**: 키워드 검색 (의미론적/벡터 검색 아님)
- **샘플 데이터**: Contoso Travel 서비스에 대한 5개 문서

**샘플 문서**:
1. 이국적인 목적지로 가는 럭셔리 휴가 패키지
2. 맞춤형 일정 계획이 포함된 프리미엄 여행 서비스
3. 여행 보험 커버리지 세부 사항
4. 인기 목적지 (몰디브, 스위스 알프스, 아프리카 사파리)
5. 부티크 호텔 및 가이드 투어에 대한 독점 접근

### 2. Semantic Kernel 에이전트

**목적**: 대화 흐름과 함수 실행 조율

```python
agent = ChatCompletionAgent(
    service=chat_completion_service,          # OpenAI 서비스
    plugins=[SearchPlugin(search_client), WeatherInfoPlugin()],  # 사용 가능한 도구
    name="TravelAgent",                       # 에이전트 식별자
    instructions="제공된 도구와 컨텍스트를 사용하여 여행 쿼리에 답변합니다..."
)
```

**구성**:
- **서비스**: AsyncOpenAI 클라이언트 (GitHub Models 추론 엔드포인트)
- **모델**: `gpt-4o-mini` (비용 효율적, 스트리밍 지원)
- **플러그인**: LLM이 사용 가능한 도구로 등록됨
- **명령어**: 환각을 방지하기 위한 시스템 수준의 지침

### 3. LLM 서비스 구성

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

**기능**:
- 비차단 I/O를 위한 비동기 클라이언트
- 비용 효율적인 추론을 위한 GitHub Models 통합
- 기본적으로 스트리밍 지원 활성화

---

## 🔌 플러그인 아키텍처

### SearchPlugin (RAG 핵심)

**책임**: 문서 검색 및 컨텍스트 증강

#### 함수 1: `retrieve_documents`

```python
@kernel_function(
    name="retrieve_documents",
    description="Azure Search 서비스에서 문서를 검색합니다."
)
def get_retrieval_context(self, query: str) -> str:
    results = self.search_client.search(query)
    context_strings = [
        f"Document: {result['content']}"
        for result in results
    ]
    return "\\n\\n".join(context_strings) if context_strings else "결과 없음"
```

**작동 방식**:
1. 문자열로 사용자 쿼리 받음
2. Azure Search 인덱스에 대해 키워드 검색 실행
3. 결과를 "Document: {content}" 문자열로 포맷
4. 여러 결과를 줄 바꿈으로 조인
5. 포맷된 컨텍스트 또는 "결과 없음" 반환

**LLM 결정 지점**: LLM은 문서 검색의 이점이 있을 수 있는 질문을 할 때 자동으로 이 함수를 호출합니다.

#### 함수 2: `build_augmented_prompt`

```python
@kernel_function(
    name="build_augmented_prompt",
    description="검색 컨텍스트 또는 함수 결과를 사용하여 증강된 프롬프트를 만듭니다."
)
def build_augmented_prompt(self, query: str, retrieval_context: str) -> str:
    return (
        f"검색된 컨텍스트:\\n{retrieval_context}\\n\\n"
        f"사용자 쿼리: {query}\\n\\n"
        "먼저 검색된 컨텍스트를 검토하십시오. 이것이 쿼리에 답하지 않으면, "
        "답변을 제공할 수 있는 사용 가능한 플러그인 함수를 호출해 보세요. "
        "사용 가능한 컨텍스트가 없으면 그렇게 말하세요."
    )
```

**목적**:
- 검색 컨텍스트를 통한 사용자 쿼리 증강 구조화
- 폴백 동작에 대한 지침 제공
- 초기 컨텍스트가 충분하지 않으면 다른 함수를 시도하도록 LLM 안내

**참고**: 정의되었지만 시연된 흐름에서 직접 호출되지 않음

### WeatherInfoPlugin (보조 데이터)

**책임**: 여행 목적지에 대한 온도 정보 제공

```python
class WeatherInfoPlugin:
    def __init__(self):
        self.destination_temperatures = {
            "maldives": "82°F (28°C)",
            "swiss alps": "45°F (7°C)",
            "african safaris": "75°F (24°C)"
        }

    @kernel_function(
        description="특정 여행 목적지의 평균 온도를 가져옵니다."
    )
    def get_destination_temperature(self, destination: str) -> Annotated[str, "목적지의 평균 온도를 반환합니다."]:
        normalized_destination = destination.lower()

        if normalized_destination in self.destination_temperatures:
            return f"{destination}의 평균 온도는 {self.destination_temperatures[normalized_destination]}입니다."
        else:
            return f"죄송하지만, {destination}에 대한 온도 정보가 없습니다. 사용 가능한 목적지: 몰디브, 스위스 알프스, 아프리카 사파리"
```

**기능**:
- 대소문자를 구분하지 않는 목적지 매칭
- 사용 가능한 옵션이 포함된 우아한 폴백
- 하드코딩된 데이터 (실제 API 호출로 교체 가능)
- 설명서를 위한 반환 타입 주석

---

## 🔄 실행 흐름

### 전체 쿼리 처리

```
사용자 입력
    ↓
[에이전트가 invoke_stream()을 통해 쿼리 수신]
    ↓
[LLM이 쿼리와 함수 설명 분석]
    ↓
[LLM이 호출할 함수 결정]
    ↓
[함수 실행 및 결과 수집]
    ↓
[사용자에게 응답 청크 스트리밍]
    ├── FunctionCallContent (함수 이름 + 인수)
    ├── FunctionResultContent (함수 출력)
    └── StreamingTextContent (LLM 생성 텍스트)
    ↓
[지속성을 위해 대화 스레드 업데이트]
```

### 주요 처리 패턴

#### 1. 스레드 기반 대화 관리

```python
thread: ChatHistoryAgentThread | None = None

async for response in agent.invoke_stream(
    messages=user_input,
    thread=thread,
):
    thread = response.thread  # 대화 기록 유지
    # 응답 항목 처리
```

**이점**:
- 여러 턴 간 컨텍스트 보존
- 후속 질문 활성화
- 대화 세션당 단일 스레드

#### 2. 스트리밍 콘텐츠 처리

`invoke_stream()` 메서드는 여러 콘텐츠 타입을 포함하는 응답을 생성합니다:

```python
async for response in agent.invoke_stream(messages=user_input, thread=thread):
    content_items = list(response.items)

    for item in content_items:
        if isinstance(item, FunctionCallContent):
            # 함수 호출 처리
        elif isinstance(item, FunctionResultContent):
            # 함수 결과 처리
        elif isinstance(item, StreamingTextContent):
            # LLM 텍스트 청크 처리
```

---

## 📡 스트리밍 구현 세부 사항

### 3단계 응답 처리

#### 단계 1: 함수 호출 버퍼링

```python
current_function_name = None
argument_buffer = ""

if isinstance(item, FunctionCallContent):
    if item.function_name:
        current_function_name = item.function_name

    # 인수가 청크로 스트림됨 - 축적
    if isinstance(item.arguments, str):
        argument_buffer += item.arguments
```

**버퍼링이 필요한 이유?**
- 함수 인수가 JSON 조각으로 스트림됨
- 파싱하기 전에 완전한 JSON을 축적해야 함
- 강력한 인수 재구성 활성화

#### 단계 2: 함수 결과 처리

```python
elif isinstance(item, FunctionResultContent):
    # 대기 중인 함수 호출 완료
    if current_function_name:
        formatted_args = argument_buffer.strip()
        try:
            parsed_args = json.loads(formatted_args)
            formatted_args = json.dumps(parsed_args)  # 예쁘게 인쇄
        except Exception:
            pass  # 원시 문자열로 폴백

        function_calls.append(
            f"함수 호출: {current_function_name}({formatted_args})"
        )
        current_function_name = None
        argument_buffer = ""

    function_calls.append(f"\\n함수 결과:\\n\\n{item.result}")
```

**오류 처리**:
- JSON 파싱 실패 시 우아한 폴백
- 잘못된 인수가 있어도 함수 호출 투명성 유지

#### 단계 3: 텍스트 축적

```python
elif isinstance(item, StreamingTextContent) and item.text:
    full_response.append(item.text)
```

**결과**: `full_response`는 완전한 LLM 응답을 구성하는 텍스트 청크 목록

### UI 렌더링

```python
html_output = (
    "<div style='margin-bottom:10px'>"
    "<details>"
    "<summary style='cursor:pointer; font-weight:bold; color:#0066cc;'>"
    "함수 호출 (확장하려면 클릭)</summary>"
    "<div style='margin:10px; padding:10px; background-color:#f8f8f8; "
    "border:1px solid #ddd; border-radius:4px; white-space:pre-wrap;'>"
    f"{chr(10).join(function_calls)}"
    "</div></details></div>"
)
```

**기능**:
- 접을 수 있는 함수 호출 세부 정보
- `white-space:pre-wrap`으로 형식 보존
- "뒷단계"를 사용자 대면 응답과 분리

---

## 💡 상호 작용 예제

### 쿼리 1: 문서 검색

**사용자**: "Contoso의 여행 보험 커버리지를 설명해 줄 수 있습니까?"

**실행**:
```
1. LLM이 "보험 커버리지"에 대한 쿼리 확인
2. LLM 호출: retrieve_documents(query="Contoso 여행 보험 커버리지")
3. Azure Search가 4개의 관련 문서 반환:
   - "Contoso의 여행 보험은 의료 응급 상황, 여행 취소, 분실 수하물을 보장합니다."
   - "Contoso Travel은 전 세계 이국적인 목적지로의 럭셔리 휴가 패키지를 제공합니다."
   - "Contoso Travel은 부티크 호텔 및 비공개 가이드 투어에 대한 독점 접근을 제공합니다."
   - "당사의 프리미엄 여행 서비스에는 맞춤형 일정 계획 및 24/7 컨시어주 지원이 포함됩니다."
4. LLM이 응답을 합성:
   - 의료 응급 상황 커버리지
   - 여행 취소 보호
   - 분실 수하물 보상
```

**에이전트 응답**:
```
Contoso의 여행 보험 커버리지는 다음을 포함합니다:

1. **의료 응급 상황**: 여행 중 예상치 못한 의료 문제에 대한 커버리지.
2. **여행 취소**: 여행을 취소해야 하는 경우 보호.
3. **분실 수하물**: 여행 중 분실된 수하물에 대한 보상.

더 자세한 정보가 필요하면 Contoso에 직접 연락하거나 공식 문서를 참고하세요.
```

### 쿼리 2: 보조 데이터

**사용자**: "몰디브의 평균 온도는 얼마입니까?"

**실행**:
```
1. LLM이 "온도"에 대한 쿼리 확인
2. LLM 호출: get_destination_temperature(destination="Maldives")
3. WeatherInfoPlugin 반환: "몰디브의 평균 온도는 82°F (28°C)입니다."
4. LLM이 온도 정보로 응답
```

### 쿼리 3: 다중 함수 조율

**사용자**: "Contoso에서 제공하는 좋은 추운 목적지는 무엇이며 평균 온도는 얼마입니까?"

**실행**:
```
1. LLM이 검색과 온도 정보 모두 필요함을 인식
2. 첫 번째 호출: retrieve_documents(query="추운 목적지 스위스 알프스")
   → 반환: "인기 목적지는 몰디브, 스위스 알프스, 아프리카 사파리입니다."
3. 두 번째 호출: get_destination_temperature(destination="Swiss Alps")
   → 반환: "스위스 알프스의 평균 온도는 45°F (7°C)입니다."
4. LLM이 두 결과를 응답에 결합:
   - 스위스 알프스는 Contoso 목적지
   - 평균 온도는 45°F (7°C)
   - 추운 날씨 여행에 적합
```

---

## ✅ 시연된 모범 사례

### 1. 멱등성 초기화

```python
try:
    existing_index = index_client.get_index(index_name)
    print(f"인덱스 '{index_name}'이 이미 존재합니다. 기존 인덱스를 사용합니다.")
except Exception:
    print(f"새 인덱스 '{index_name}' 생성 중...")
    index_client.create_index(index)
```

**이점**: 오류 없이 여러 번 안전하게 실행 가능

### 2. 방어적 오류 처리

```python
try:
    parsed_args = json.loads(formatted_args)
    formatted_args = json.dumps(parsed_args)
except Exception:
    pass  # 원시 문자열로 폴백
```

**이점**: JSON 파싱 실패 시 우아한 성능 저하

### 3. 명확한 함수 설명

```python
@kernel_function(
    name="retrieve_documents",
    description="Azure Search 서비스에서 문서를 검색합니다."
)
```

**이점**: 설명적인 이름은 LLM이 올바른 함수 호출을 하도록 안내

### 4. 모듈식 플러그인 설계

- **SearchPlugin**: 모든 검색 논리 캡슐화
- **WeatherInfoPlugin**: 날씨 데이터에 대한 별도 관심사
- **에이전트**: 구현에 대한 결합 없이 조율

**이점**: 플러그인을 쉽게 추가, 제거 또는 수정 가능

### 5. 스트리밍 UX

- 실시간 응답 전달
- 접을 수 있는 세부 사항이 포함된 함수 호출 투명성
- 더 나은 인지된 성능

### 6. 스레드 기반 대화

- 단일 스레드 객체가 기록을 유지
- 다중 턴 대화 활성화
- 자동으로 컨텍스트 보존

---

## ⚠️ 제한 사항 및 개선 기회

### 현재 제한 사항

| 제한 사항 | 영향 | 해결책 |
|----------|------|------|
| **키워드 검색만** | 의미론적으로 유사하지만 어휘적으로 다른 콘텐츠 누락 | 의미 임베딩으로 벡터 검색 구현 |
| **하드코딩된 날씨 데이터** | 3개 목적지로 제한, 수동 업데이트 필요 | 실제 날씨 API 통합 (OpenWeather, Weather API) |
| **재순위 없음** | 검색 결과가 관련성별로 점수 매김되지 않음 | BM25 또는 의미 유사성 점수 추가 |
| **제한된 오류 처리** | 검색 실패가 흐름을 중단할 수 있음 | 시도/포착 및 폴백 전략 추가 |
| **`build_augmented_prompt` 미사용** | 함수 정의되었지만 호출되지 않음 | 주 검색 흐름에 통합 |
| **간단한 스키마** | ID + 콘텐츠 필드만 | 메타데이터 필드 추가 (날짜, 저자, 소스 등) |

### 개선 기회

#### 1. 벡터 검색 구현
```python
# 현재: 키워드 검색
results = self.search_client.search(query)

# 개선됨: 임베딩을 사용한 의미 검색
vector = embeddings_service.embed(query)
results = self.search_client.search(
    query=None,
    vector=vector,
    k=5,
    vectors_query_kind="similarity"
)
```

#### 2. 실제 날씨 통합 추가
```python
import aiohttp

async def get_destination_temperature(self, destination: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://api.openweathermap.org/data/2.5/weather?q={destination}"
        ) as resp:
            data = await resp.json()
            return f"{destination}의 온도: {data['main']['temp']}°C"
```

#### 3. 인덱스 스키마 개선
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

#### 4. 결과 재순위 추가
```python
def get_retrieval_context(self, query: str) -> str:
    results = self.search_client.search(query)

    # 관련성 점수별로 재순위 지정
    ranked = sorted(
        results,
        key=lambda x: self._calculate_relevance(query, x['content']),
        reverse=True
    )[:5]

    return format_results(ranked)
```

#### 5. `build_augmented_prompt` 통합
```python
async for response in agent.invoke_stream(messages=user_input, thread=thread):
    # 검색 후 build_augmented_prompt 사용
    retrieval_context = search_plugin.get_retrieval_context(user_input)
    augmented_prompt = search_plugin.build_augmented_prompt(
        user_input,
        retrieval_context
    )
```

---

## 🎯 핵심 요점

1. **RAG 아키텍처**: 근거 기반 응답을 위해 검색과 생성 결합
2. **함수 호출**: Semantic Kernel이 자동으로 플러그인 함수를 LLM 호출 가능 도구로 노출
3. **다중 플러그인 설계**: 다양한 관심사(검색, 날씨 등)에 대한 별도 플러그인
4. **스트리밍 응답**: 함수 호출의 전체 투명성과 함께 실시간 전달
5. **대화 메모리**: 스레드 기반 컨텍스트가 다중 턴 상호 작용 보존
6. **모듈식 & 확장 가능**: 새 플러그인을 쉽게 추가하거나 데이터 소스 교체 가능

---

## 📚 프로덕션 준비

이 노트북은 여러 **프로덕션 준비 패턴**을 시연합니다:

✅ 멱등성 리소스 초기화
✅ 우아한 오류 처리
✅ 더 나은 UX를 위한 스트리밍
✅ 명확한 관심사의 분리
✅ 투명한 디버깅 기능
✅ 다중 턴 대화 지원

**다음 없이는 프로덕션 준비가 완료되지 않음**:
- 의미 검색을 위한 벡터 임베딩
- 실제 외부 API 통합
- 포괄적인 오류 처리
- 성능을 위한 결과 캐싱
- 속도 제한 및 회로 차단기
- 포괄적인 로깅 및 모니터링

---

## 🔗 관련 개념

- **검색 증강 생성(RAG)**: 사실 정확성을 위해 검색과 생성 결합
- **함수 호출**: 함수 설명을 기반으로 LLM 기반 함수 호출
- **Semantic Kernel**: LLM 기반 에이전트 구축을 위한 Microsoft 프레임워크
- **Azure AI Search**: 문서 색인 및 검색을 위한 관리 검색 서비스
- **스트리밍 응답**: 더 나은 UX를 위한 실시간 토큰 전달
- **에이전트 아키텍처**: 도구를 호출하고 컨텍스트를 유지할 수 있는 자율 시스템

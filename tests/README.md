# Testing Context Engine

## Running Tests

```bash
# Install dependencies (including dev/test group)
uv sync --group dev

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run specific test file
pytest tests/agents/test_agents.py -v

# Run specific test
pytest tests/agents/test_agents.py::TestLibrarianAgent::test_initialization -v
```

## Test Structure

```
tests/
├── agents/                  # Agent unit tests
├── retrieval/               # Retrieval tests
├── runtime/                 # Engine + API integration tests
├── conftest.py              # Shared fixtures
└── README.md
```

## Test Coverage

Current tests cover:

- ✅ Agent initialization
- ✅ Execution without dependencies (graceful fallback)
- ✅ Input validation
- ✅ Output format
- ✅ Evidence item creation
- ✅ Pinecone match conversion

## Next Steps

1. **Add integration tests**: Test full pipeline with mocked Pinecone
2. **Add retrieval tests**: Test PineconeRetriever and LLMReranker
3. **Add security tests**: Test prompt injection detection
4. **Add performance tests**: Benchmark agent execution times

## Running Tests in CI/CD

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    uv sync --group dev
    pytest tests/ --cov=app --cov-report=xml
```

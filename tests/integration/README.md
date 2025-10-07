# Integration Tests - Lexora Agentic RAG SDK

This directory contains end-to-end integration tests for the Lexora Agentic RAG SDK.

## Test Files

### test_e2e_simple.py
Simplified integration tests that can run standalone without pytest. These tests verify:

1. **Complete RAG Pipeline** - Tests the full workflow from corpus creation to querying
2. **Multi-Corpus Workflow** - Tests managing multiple corpora simultaneously
3. **Error Handling** - Tests error scenarios and recovery mechanisms
4. **FAISS Persistence** - Tests data persistence in the FAISS backend
5. **Large Document Batch** - Tests handling of large document batches (50+ documents)

### test_e2e_workflows.py
Comprehensive pytest-based integration tests with fixtures and detailed test scenarios.

## Running the Tests

### Standalone Execution
```bash
python tests/integration/test_e2e_simple.py
```

### With Pytest (if installed)
```bash
pytest tests/integration/test_e2e_workflows.py -v
```

## Test Coverage

The integration tests cover the following requirements:
- **Requirement 2.1**: Query processing workflows
- **Requirement 2.4**: Multi-step reasoning scenarios
- **Requirement 4.3**: Vector database persistence
- **Requirement 5.1**: Document processing
- **Requirement 6.1**: Complete RAG workflows
- **Requirement 6.4**: Context management
- **Requirement 7.2**: Error handling and recovery

## Test Results

All 5 integration tests pass successfully:
- ✅ Complete RAG Pipeline
- ✅ Multi-Corpus Workflow
- ✅ Error Handling
- ✅ FAISS Persistence
- ✅ Large Document Batch

## Notes

- Tests use the FAISS backend by default
- Mock embeddings are used for testing (no API keys required)
- Tests automatically clean up created corpora
- Some old corpus files may show warnings during loading (can be ignored)

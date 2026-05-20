# IAbel Project Cleanup Recommendations

## 🎯 Goal: Simplify architecture for better maintainability and debugging (especially streaming issues)

## 📋 CLEANUP CHECKLIST

### 1. DELETE Redundant Backend Services
```bash
# These are replaced by enhanced RAG system
rm backend/app/services/chat_service.py
rm backend/app/services/embedding_service.py  
rm backend/app/services/pdf_processor.py
```

### 2. DELETE Fine-tuning Complexity (95% of files)
```bash
# Keep only hybrid_rag_lora.py for integration
rm -rf backend/fine_tuning/outputs/          # ~2GB of checkpoints
rm backend/fine_tuning/train_*.py           # 9 training scripts
rm backend/fine_tuning/test_*.py            # 4 test files
rm backend/fine_tuning/data_processor.py
rm backend/fine_tuning/lora_trainer.py
rm backend/fine_tuning/*.log
```

### 3. DELETE Root-level Utilities
```bash
rm demo_enhanced_rag.py
rm iabel_local.py
rm setup_*.py
rm backend/*.py                              # 20+ utility scripts
```

### 4. DELETE Test Files (Create proper test structure later)
```bash
rm backend/test_*.py                         # 4 test files
```

### 5. DELETE Redundant Components in local_rag/
```bash
# These might be over-engineered for simple use case
rm -rf local_rag/caching/                   # Replaced by redis cache
rm -rf local_rag/chunking/                  # Basic chunking sufficient
rm -rf local_rag/reranking/                 # Advanced feature
rm -rf local_rag/quality/                   # Advanced feature
rm -rf local_rag/fusion/                    # Advanced feature
rm -rf local_rag/learning/                  # Advanced feature
rm -rf local_rag/citations/                 # Advanced feature
rm -rf local_rag/compression/               # Advanced feature
rm -rf local_rag/memory/                    # Advanced feature
```

### 6. CONSOLIDATE Requirements
```bash
# Multiple requirements files - keep only one
rm backend/requirements_*.txt               # Keep only requirements.txt
rm requirements_*.txt                       # Root level duplicates
```

## 🏗️ SIMPLIFIED ARCHITECTURE

### Core Components (Keep):
1. **backend/app/main.py** - FastAPI server
2. **backend/app/services/rag_service.py** - Main service
3. **local_rag/enhanced_rag_system.py** - RAG logic
4. **local_rag/models/ollama_client.py** - LLM client
5. **local_rag/vectorstore/** - Vector database
6. **local_rag/embeddings/** - Embeddings
7. **local_rag/processors/** - PDF processing
8. **frontend/** - React app

### Benefits of Cleanup:
- ✅ **Easier debugging** - Clear data flow
- ✅ **Faster development** - Less cognitive overhead  
- ✅ **Better performance** - Remove unused imports/processing
- ✅ **Simpler deployment** - Fewer dependencies
- ✅ **Fix streaming issues** - Clearer code path

## 🚦 EXECUTION PLAN

### Phase 1: Backup & Delete (Safe)
```bash
# Create backup
cp -r backend backend_backup
cp -r local_rag local_rag_backup

# Delete obvious unused files
rm -rf backend/fine_tuning/outputs/
rm backend/fine_tuning/train_*.py
rm backend/test_*.py
rm demo_*.py
rm iabel_local.py
```

### Phase 2: Test Core Functionality
```bash
# Start backend
cd backend && python app/main.py

# Test endpoints
curl http://localhost:8000/health/
curl http://localhost:8000/status/

# Start frontend  
cd frontend && npm start
```

### Phase 3: Advanced Cleanup (After testing)
```bash
# Remove over-engineered components
rm -rf local_rag/caching/
rm -rf local_rag/chunking/
# ... etc
```

## 🎯 Expected Result:
- **From 38k+ files to ~50 core files**
- **Clear single data flow path**
- **Easier to debug streaming issues**
- **Maintainable codebase**

## ⚠️ Risks & Mitigation:
- **Risk:** Breaking functionality
- **Mitigation:** Keep backups, test after each phase
- **Risk:** Missing dependencies  
- **Mitigation:** Test all endpoints before cleanup
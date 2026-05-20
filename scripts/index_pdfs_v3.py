"""
Script to index PDFs into RAG v3 system
Extracts author/year metadata and creates academic citations
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
env_path = project_root / "backend" / ".env"
load_dotenv(env_path)
print(f"📝 Loaded .env from: {env_path}")

from local_rag.rag_v3_system import RAGv3System


def main():
    print("=" * 60)
    print("RAG v3 PDF Indexing Script")
    print("English Responses + Academic Citations (Author, Year)")
    print("=" * 60)

    # Paths
    pdf_directory = project_root / "backend" / "data" / "pdfs"
    vectorstore_path = project_root / "backend" / "data" / "vectorstore_v3"

    if not pdf_directory.exists():
        print(f"\n❌ PDF directory not found: {pdf_directory}")
        print("Please ensure PDFs are in backend/data/pdfs/")
        return False

    print(f"\n📁 PDF Directory: {pdf_directory}")
    print(f"💾 Vector Store: {vectorstore_path}")

    # Initialize RAG v3 system
    print("\n🚀 Initializing RAG v3 System...")
    rag_v3 = RAGv3System(
        vectorstore_path=str(vectorstore_path),
        embedder_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        chunk_size=500,
        chunk_overlap=80
    )

    # Index PDFs
    print("\n📄 Starting PDF indexing...")
    print("-" * 60)

    success = rag_v3.index_pdf_directory(
        pdf_directory=str(pdf_directory),
        force_reindex=True  # Reindex all PDFs
    )

    if success:
        print("\n" + "=" * 60)
        print("✅ INDEXING COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        # Show system status
        status = rag_v3.get_system_status()
        print("\n📊 System Status:")
        print(f"   Version: {status.get('version', 'unknown')}")

        if 'error' in status:
            print(f"   ⚠️ Status Error: {status['error']}")

        if 'vector_store' in status:
            print(f"   Total Documents: {status['vector_store']['total_documents']}")

        if 'features' in status:
            print(f"   Language: {status['features']['language']}")
            print(f"   Citation Format: {status['features']['citation_format']}")

        if 'llm' in status:
            print(f"   LLM: {status['llm']['provider']} - {status['llm']['model']}")

        print("\n✨ RAG v3 is ready to use!")
        print("   - English responses")
        print("   - Academic citations: (Author, Year)")
        print("   - No reference boxes in UI")

        return True
    else:
        print("\n" + "=" * 60)
        print("❌ INDEXING FAILED")
        print("=" * 60)
        print("Please check error messages above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

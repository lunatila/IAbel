#!/usr/bin/env python3
"""
Quick test script to verify LoRA setup is working
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test all critical imports"""
    logger.info("Testing imports...")
    
    success = []
    failures = []
    
    # Test PDF extractors
    try:
        import PyMuPDF
        success.append("PyMuPDF")
    except ImportError:
        failures.append("PyMuPDF")
    
    try:
        import pdfplumber
        success.append("pdfplumber")
    except ImportError:
        failures.append("pdfplumber")
    
    # Test ML libraries
    ml_libs = ['transformers', 'torch', 'datasets', 'numpy', 'pandas']
    for lib in ml_libs:
        try:
            __import__(lib)
            success.append(lib)
        except ImportError:
            failures.append(lib)
    
    # Test LoRA specific
    try:
        import peft
        success.append("peft")
    except ImportError:
        failures.append("peft")
    
    try:
        import accelerate
        success.append("accelerate")
    except ImportError:
        failures.append("accelerate")
    
    logger.info(f"✅ Available: {success}")
    if failures:
        logger.warning(f"❌ Missing: {failures}")
    
    return len(failures) == 0

def test_data_processor():
    """Test data processor initialization"""
    logger.info("Testing data processor...")
    
    try:
        from data_processor import AcademicDataProcessor
        
        processor = AcademicDataProcessor(
            chunk_size=512,
            min_technical_density=0.1
        )
        
        logger.info("✅ Data processor initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Data processor failed: {e}")
        return False

def test_pdf_extraction():
    """Test PDF extraction with available PDFs"""
    logger.info("Testing PDF extraction...")
    
    try:
        from data_processor import AcademicDataProcessor
        
        processor = AcademicDataProcessor(chunk_size=256, min_technical_density=0.1)
        
        # Look for PDFs
        pdf_dir = Path("../data/pdfs")
        if not pdf_dir.exists():
            logger.warning(f"PDF directory not found: {pdf_dir}")
            return False
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDF files found for testing")
            return False
        
        # Test with smallest PDF
        test_pdf = min(pdf_files, key=lambda x: x.stat().st_size)
        logger.info(f"Testing with: {test_pdf.name} ({test_pdf.stat().st_size / 1024:.1f} KB)")
        
        chunks = processor.extract_text_from_pdf(str(test_pdf))
        
        if chunks:
            logger.info(f"✅ Extracted {len(chunks)} chunks from test PDF")
            
            # Show sample chunk
            if chunks[0].text:
                sample_text = chunks[0].text[:100] + "..." if len(chunks[0].text) > 100 else chunks[0].text
                logger.info(f"Sample text: {sample_text}")
            
            return True
        else:
            logger.warning("No chunks extracted from test PDF")
            return False
            
    except Exception as e:
        logger.error(f"❌ PDF extraction failed: {e}")
        return False

def test_gpu_availability():
    """Test GPU/CUDA availability"""
    logger.info("Testing GPU availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f"✅ GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            return True
        else:
            logger.warning("⚠️ No CUDA GPU detected. Training will use CPU.")
            return False
    except Exception as e:
        logger.error(f"❌ GPU test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🔍 IAbel LoRA Setup Test")
    logger.info("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Processor Test", test_data_processor), 
        ("PDF Extraction Test", test_pdf_extraction),
        ("GPU Test", test_gpu_availability)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 {test_name}")
        logger.info("-" * 30)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 40)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 40)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} {test_name}")
        if success:
            passed += 1
    
    logger.info(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! Ready for LoRA training.")
        return True
    elif passed >= len(results) - 1:
        logger.info("⚠️ Most tests passed. Training should work with minor issues.")
        return True
    else:
        logger.info("❌ Multiple issues detected. Fix dependencies before training.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Enhanced RAG Setup Script
Automatically configures and installs the enhanced IAbel RAG system
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import shutil
from typing import List, Dict, Any
import json

class EnhancedRAGSetup:
    """
    Setup and configuration for Enhanced RAG System
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / "venv"
        self.backend_path = self.project_root / "backend"
        self.local_rag_path = self.project_root / "local_rag"
        
        self.python_executable = sys.executable
        self.platform = platform.system().lower()
        
        self.setup_log = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log setup messages"""
        log_entry = f"[{level}] {message}"
        print(log_entry)
        self.setup_log.append(log_entry)
    
    def run_command(self, command: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run shell command with logging"""
        cmd_str = " ".join(command)
        self.log(f"Running: {cmd_str}")
        
        try:
            result = subprocess.run(
                command,
                check=check,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                self.log(f"✅ Command successful: {cmd_str}")
            else:
                self.log(f"❌ Command failed: {cmd_str}", "ERROR")
                self.log(f"Error output: {result.stderr}", "ERROR")
            
            return result
            
        except subprocess.CalledProcessError as e:
            self.log(f"❌ Command failed with exception: {cmd_str}", "ERROR")
            self.log(f"Error: {e}", "ERROR")
            raise
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        version = sys.version_info
        
        if version.major == 3 and version.minor >= 9:
            self.log(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
            return True
        else:
            self.log(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible", "ERROR")
            self.log("   Enhanced RAG requires Python 3.9+", "ERROR")
            return False
    
    def create_virtual_environment(self) -> bool:
        """Create virtual environment if it doesn't exist"""
        if self.venv_path.exists():
            self.log("📁 Virtual environment already exists")
            return True
        
        try:
            self.log("🔧 Creating virtual environment...")
            self.run_command([self.python_executable, "-m", "venv", str(self.venv_path)])
            self.log("✅ Virtual environment created")
            return True
        except Exception as e:
            self.log(f"❌ Failed to create virtual environment: {e}", "ERROR")
            return False
    
    def get_venv_python(self) -> str:
        """Get path to virtual environment Python"""
        if self.platform == "windows":
            return str(self.venv_path / "Scripts" / "python.exe")
        else:
            return str(self.venv_path / "bin" / "python")
    
    def get_venv_pip(self) -> str:
        """Get path to virtual environment pip"""
        if self.platform == "windows":
            return str(self.venv_path / "Scripts" / "pip.exe")
        else:
            return str(self.venv_path / "bin" / "pip")
    
    def upgrade_pip(self) -> bool:
        """Upgrade pip to latest version"""
        try:
            self.log("⬆️ Upgrading pip...")
            pip_path = self.get_venv_pip()
            self.run_command([pip_path, "install", "--upgrade", "pip"])
            self.log("✅ Pip upgraded")
            return True
        except Exception as e:
            self.log(f"⚠️ Pip upgrade failed: {e}", "WARN")
            return False
    
    def install_requirements(self) -> bool:
        """Install enhanced requirements"""
        requirements_file = self.project_root / "requirements_enhanced.txt"
        
        if not requirements_file.exists():
            self.log("❌ requirements_enhanced.txt not found", "ERROR")
            return False
        
        try:
            self.log("📦 Installing enhanced requirements...")
            pip_path = self.get_venv_pip()
            
            # Install with timeout for large downloads
            self.run_command([
                pip_path, "install", "-r", str(requirements_file),
                "--timeout", "300"
            ])
            
            self.log("✅ Enhanced requirements installed")
            return True
        except Exception as e:
            self.log(f"❌ Failed to install requirements: {e}", "ERROR")
            return False
    
    def install_spacy_models(self) -> bool:
        """Install spaCy language models"""
        python_path = self.get_venv_python()
        
        models = ["pt_core_news_sm", "en_core_web_sm"]
        success_count = 0
        
        for model in models:
            try:
                self.log(f"📚 Installing spaCy model: {model}")
                self.run_command([
                    python_path, "-m", "spacy", "download", model
                ], check=False)  # Don't fail if one model fails
                success_count += 1
            except Exception as e:
                self.log(f"⚠️ Failed to install {model}: {e}", "WARN")
        
        if success_count > 0:
            self.log(f"✅ Installed {success_count}/{len(models)} spaCy models")
            return True
        else:
            self.log("❌ Failed to install any spaCy models", "ERROR")
            return False
    
    def check_redis_availability(self) -> Dict[str, Any]:
        """Check if Redis is available"""
        try:
            import redis
            
            # Try to connect to Redis
            r = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=2)
            r.ping()
            
            info = r.info()
            
            return {
                'available': True,
                'version': info.get('redis_version', 'unknown'),
                'memory': info.get('used_memory_human', 'unknown')
            }
            
        except ImportError:
            return {
                'available': False,
                'error': 'redis package not installed'
            }
        except Exception as e:
            return {
                'available': False,
                'error': str(e)
            }
    
    def check_ollama_availability(self) -> Dict[str, Any]:
        """Check if Ollama is available"""
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                version = result.stdout.strip()
                
                # Check available models
                models_result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                models = []
                if models_result.returncode == 0:
                    lines = models_result.stdout.strip().split('\n')[1:]  # Skip header
                    models = [line.split()[0] for line in lines if line.strip()]
                
                return {
                    'available': True,
                    'version': version,
                    'models': models
                }
            else:
                return {
                    'available': False,
                    'error': 'ollama command failed'
                }
                
        except FileNotFoundError:
            return {
                'available': False,
                'error': 'ollama not found in PATH'
            }
        except Exception as e:
            return {
                'available': False,
                'error': str(e)
            }
    
    def create_directories(self) -> bool:
        """Create necessary directories"""
        directories = [
            self.local_rag_path / "reranking",
            self.local_rag_path / "fusion", 
            self.local_rag_path / "chunking",
            self.local_rag_path / "caching",
            self.local_rag_path / "quality",
            self.backend_path / "data" / "pdfs",
            self.backend_path / "data" / "vectorstore_enhanced",
            self.backend_path / "logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.log(f"📁 Created directory: {directory.relative_to(self.project_root)}")
        
        return True
    
    def create_init_files(self) -> bool:
        """Create __init__.py files for packages"""
        init_files = [
            self.local_rag_path / "__init__.py",
            self.local_rag_path / "reranking" / "__init__.py",
            self.local_rag_path / "fusion" / "__init__.py",
            self.local_rag_path / "chunking" / "__init__.py",
            self.local_rag_path / "caching" / "__init__.py",
            self.local_rag_path / "quality" / "__init__.py"
        ]
        
        for init_file in init_files:
            if not init_file.exists():
                init_file.write_text('"""Enhanced RAG Component"""')
                self.log(f"📄 Created __init__.py: {init_file.relative_to(self.project_root)}")
        
        return True
    
    def test_installation(self) -> Dict[str, Any]:
        """Test the enhanced installation"""
        python_path = self.get_venv_python()
        
        test_results = {
            'basic_imports': False,
            'enhanced_imports': False,
            'models_accessible': False,
            'system_check': {}
        }
        
        # Test basic imports
        test_code = '''
import sys
sys.path.insert(0, ".")

try:
    import torch
    import sentence_transformers
    import chromadb
    import redis
    print("BASIC_IMPORTS:OK")
except Exception as e:
    print(f"BASIC_IMPORTS:ERROR:{e}")

try:
    from local_rag.enhanced_rag_system import EnhancedRAGSystem
    from local_rag.reranking.cross_encoder_reranker import get_reranker
    from local_rag.fusion.rag_fusion import get_rag_fusion
    print("ENHANCED_IMPORTS:OK")
except Exception as e:
    print(f"ENHANCED_IMPORTS:ERROR:{e}")

try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("MODELS_ACCESSIBLE:OK")
except Exception as e:
    print(f"MODELS_ACCESSIBLE:ERROR:{e}")
'''
        
        try:
            result = subprocess.run(
                [python_path, "-c", test_code],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=self.project_root
            )
            
            for line in result.stdout.strip().split('\n'):
                if line.startswith('BASIC_IMPORTS:'):
                    test_results['basic_imports'] = line.endswith(':OK')
                elif line.startswith('ENHANCED_IMPORTS:'):
                    test_results['enhanced_imports'] = line.endswith(':OK')
                elif line.startswith('MODELS_ACCESSIBLE:'):
                    test_results['models_accessible'] = line.endswith(':OK')
            
        except Exception as e:
            self.log(f"❌ Installation test failed: {e}", "ERROR")
        
        # System checks
        test_results['system_check'] = {
            'redis': self.check_redis_availability(),
            'ollama': self.check_ollama_availability()
        }
        
        return test_results
    
    def generate_setup_report(self, test_results: Dict[str, Any]) -> str:
        """Generate setup completion report"""
        report = "\n" + "="*60 + "\n"
        report += "🚀 ENHANCED RAG SETUP COMPLETED\n"
        report += "="*60 + "\n\n"
        
        # Installation status
        report += "📦 INSTALLATION STATUS:\n"
        report += f"   Basic imports: {'✅' if test_results['basic_imports'] else '❌'}\n"
        report += f"   Enhanced imports: {'✅' if test_results['enhanced_imports'] else '❌'}\n"
        report += f"   Models accessible: {'✅' if test_results['models_accessible'] else '❌'}\n"
        
        # System status
        redis_info = test_results['system_check']['redis']
        ollama_info = test_results['system_check']['ollama']
        
        report += "\n🔧 SYSTEM STATUS:\n"
        report += f"   Redis: {'✅' if redis_info['available'] else '❌'}"
        
        if redis_info['available']:
            report += f" (v{redis_info.get('version', '?')})\n"
        else:
            report += f" ({redis_info['error']})\n"
        
        report += f"   Ollama: {'✅' if ollama_info['available'] else '❌'}"
        
        if ollama_info['available']:
            models = ollama_info.get('models', [])
            report += f" ({len(models)} models available)\n"
            if models:
                report += f"      Models: {', '.join(models[:3])}\n"
        else:
            report += f" ({ollama_info['error']})\n"
        
        # Enhanced features
        report += "\n🚀 ENHANCED FEATURES:\n"
        features = [
            "✅ Semantic Chunking - Intelligent document segmentation",
            "✅ RAG Fusion - Multi-query parallel retrieval",
            "✅ Cross-Encoder Re-ranking - Superior relevance scoring",
            "✅ Enhanced Caching - Redis-based performance optimization",
            "✅ Self-Critique - Automated quality validation",
            "✅ Context Enhancement - Smart definition linking",
            "✅ Technical Boosting - Domain-specific relevance"
        ]
        
        for feature in features:
            report += f"   {feature}\n"
        
        # Usage instructions
        report += "\n📋 NEXT STEPS:\n"
        
        if test_results['basic_imports'] and test_results['enhanced_imports']:
            report += "   1. ✅ All components installed successfully\n"
            report += "   2. 🚀 Start using enhanced RAG:\n"
            report += "      python local_rag/enhanced_rag_system.py\n"
            report += "   3. 📚 Add PDFs to backend/data/pdfs/\n"
            report += "   4. 🔧 Configure environment variables if needed\n"
        else:
            report += "   1. ❌ Installation incomplete - check errors above\n"
            report += "   2. 🔄 Try running setup again\n"
            report += "   3. 📝 Check logs for specific error messages\n"
        
        if not redis_info['available']:
            report += "\n⚠️  REDIS SETUP:\n"
            report += "   Redis is not available but recommended for production\n"
            report += "   Install with: sudo apt install redis-server\n"
            report += "   Or use Docker: docker run -d -p 6379:6379 redis:7-alpine\n"
        
        if not ollama_info['available']:
            report += "\n⚠️  OLLAMA SETUP:\n"
            report += "   Ollama is required for local LLM inference\n"
            report += "   Install from: https://ollama.com/\n"
            report += "   Then run: ollama pull llama3.2:3b\n"
        
        report += "\n" + "="*60 + "\n"
        
        return report
    
    def setup(self) -> bool:
        """Run complete enhanced RAG setup"""
        self.log("🚀 Starting Enhanced RAG Setup...")
        
        # Check prerequisites
        if not self.check_python_version():
            return False
        
        # Create virtual environment
        if not self.create_virtual_environment():
            return False
        
        # Upgrade pip
        self.upgrade_pip()
        
        # Install requirements
        if not self.install_requirements():
            return False
        
        # Install spaCy models
        self.install_spacy_models()
        
        # Create necessary directories
        self.create_directories()
        
        # Create __init__.py files
        self.create_init_files()
        
        # Test installation
        self.log("🧪 Testing installation...")
        test_results = self.test_installation()
        
        # Generate and display report
        report = self.generate_setup_report(test_results)
        print(report)
        
        # Save setup log
        log_file = self.project_root / "setup_enhanced_rag.log"
        with open(log_file, 'w') as f:
            f.write('\n'.join(self.setup_log))
            f.write('\n\n')
            f.write(report)
        
        self.log(f"📝 Setup log saved to: {log_file}")
        
        return test_results['basic_imports'] and test_results['enhanced_imports']


def main():
    """Main setup function"""
    setup = EnhancedRAGSetup()
    
    try:
        success = setup.setup()
        
        if success:
            print("\n🎉 Enhanced RAG setup completed successfully!")
            print("   Ready to use advanced RAG capabilities")
            return 0
        else:
            print("\n❌ Enhanced RAG setup failed")
            print("   Check setup log for details")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Setup interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
import sys
import os

VENV_PACKAGES = os.path.join(os.path.dirname(__file__), 'venv/lib/python3.12/site-packages')
sys.path.insert(0, VENV_PACKAGES)
sys.path.insert(0, os.path.dirname(__file__))

import uvicorn
uvicorn.run('app.main:app', host='0.0.0.0', port=8000, reload=False)

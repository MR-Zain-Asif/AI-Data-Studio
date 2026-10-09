import sys
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
NEW_DIR = ROOT_DIR / "new_"

# Load environment variables from both root and new_ directory
load_dotenv(ROOT_DIR / ".env")
load_dotenv(NEW_DIR / ".env")

# Ensure new_ directory is on sys.path so all imports (routers, services, models) resolve properly
if str(NEW_DIR) not in sys.path:
    sys.path.insert(0, str(NEW_DIR))

# Load app from new_/main.py without circular import conflict
import importlib.util

module_path = NEW_DIR / "main.py"
spec = importlib.util.spec_from_file_location("new_main", str(module_path))
new_main = importlib.util.module_from_spec(spec)
sys.modules["new_main"] = new_main
spec.loader.exec_module(new_main)

app = new_main.app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

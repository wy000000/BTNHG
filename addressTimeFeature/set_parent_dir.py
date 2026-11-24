from pathlib import Path
import sys

def setup_parent_dir():
	current_dir = Path(__file__).resolve().parent
	project_root = current_dir.parent

	for path in (str(project_root), str(current_dir)):
		if path not in sys.path:
			sys.path.insert(0, path)

setup_parent_dir()

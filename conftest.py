# conftest.py (repo root)
import sys
import pathlib

# add project root to sys.path so tests in subfolders can import local modules
sys.path.append(str(pathlib.Path(__file__).resolve()))

pytest_plugins = ("pytest_asyncio",)

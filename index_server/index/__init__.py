"""Index Server application."""
import os
from pathlib import Path
import flask

# Create Flask application
app = flask.Flask(__name__)

# Configuration
INDEX_DIR = Path(__file__).parent / "inverted_index"
app.config["INDEX_PATH"] = os.getenv(
    "INDEX_PATH",  # Environment variable name
    INDEX_DIR / "inverted_index_1.txt"  # Default value
)

# Import API routes
import index.api.main  # noqa: E402  pylint: disable=wrong-import-position

# Load inverted index, stopwords, and pagerank into memory
index.api.main.load_index()

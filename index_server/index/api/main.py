"""Index server REST API main module."""
import os
import math
import pathlib
import flask

# Create Flask app
app = flask.Flask(__name__)

# Global data structures
inverted_index = {}
pagerank = {}
stopwords = set()
doc_info = {}  # {docid: {term_count: int, normalization: float}}


def load_stopwords():
    """Load stopwords from file."""
    stopwords_path = pathlib.Path(__file__).parent.parent / "stopwords.txt"
    with open(stopwords_path, encoding="utf-8") as file:
        return set(line.strip() for line in file)


def load_pagerank():
    """Load pagerank scores from file."""
    pagerank_path = pathlib.Path(__file__).parent.parent / "pagerank.out"
    pagerank_dict = {}
    with open(pagerank_path, encoding="utf-8") as file:
        for line in file:
            docid, score = line.strip().split(",")
            pagerank_dict[int(docid)] = float(score)
    return pagerank_dict


def load_inverted_index():
    """Load inverted index segment from file."""
    # Determine which inverted index segment to load based on INVERTED_INDEX_SEGMENT_ID env var
    segment_id = os.environ.get("INVERTED_INDEX_SEGMENT_ID", "0")
    index_path = (
        pathlib.Path(__file__).parent.parent
        / "inverted_index"
        / f"inverted_index_{segment_id}.txt"
    )
    
    inverted_idx = {}
    doc_info_dict = {}
    
    with open(index_path, encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            # Format: term idf docid1 tf1 norm1 docid2 tf2 norm2 ...
            term = parts[0]
            idf = float(parts[1])
            
            # Parse document entries (docid, tf, norm triplets)
            doc_entries = []
            i = 2
            while i < len(parts):
                docid = int(parts[i])
                tf = int(parts[i + 1])
                norm = float(parts[i + 2])
                doc_entries.append((docid, tf, norm))
                
                # Store document normalization factor
                if docid not in doc_info_dict:
                    doc_info_dict[docid] = {"normalization": norm}
                
                i += 3
            
            inverted_idx[term] = {
                "idf": idf,
                "docs": doc_entries
            }
    
    return inverted_idx, doc_info_dict


# Load data at startup
stopwords = load_stopwords()
pagerank = load_pagerank()
inverted_index, doc_info = load_inverted_index()


@app.route("/api/v1/")
def get_api():
    """Return list of available API endpoints."""
    context = {
        "hits": "/api/v1/hits/",
        "url": "/api/v1/"
    }
    return flask.jsonify(context)


@app.route("/api/v1/hits/")
def get_hits():
    """Return search results for query."""
    # Get query parameter
    query = flask.request.args.get("q", "")
    
    # Get weight parameter (default 0.5)
    weight = flask.request.args.get("w", "0.5")
    try:
        weight = float(weight)
    except ValueError:
        weight = 0.5
    
    # Tokenize and clean query
    query_terms = query.lower().split()
    query_terms = [term for term in query_terms if term not in stopwords]
    
    # Calculate scores for each document
    doc_scores = {}
    
    for term in query_terms:
        if term in inverted_index:
            idf = inverted_index[term]["idf"]
            
            for docid, tf, norm in inverted_index[term]["docs"]:
                # Calculate tf-idf contribution
                tfidf = tf * idf
                
                # Initialize document score if not exists
                if docid not in doc_scores:
                    doc_scores[docid] = {
                        "tfidf_sum": 0.0,
                        "normalization": norm
                    }
                
                # Add to tf-idf sum
                doc_scores[docid]["tfidf_sum"] += tfidf
    
    # Compute final scores
    hits = []
    for docid, score_data in doc_scores.items():
        # Normalize tf-idf score
        tfidf_score = score_data["tfidf_sum"] / score_data["normalization"]
        
        # Get pagerank score (default to 0 if not found)
        pagerank_score = pagerank.get(docid, 0.0)
        
        # Combine scores: score = w * pagerank + (1 - w) * tfidf
        final_score = weight * pagerank_score + (1 - weight) * tfidf_score
        
        hits.append({
            "docid": docid,
            "score": final_score
        })
    
    # Sort by score (descending) and then by docid (ascending) for ties
    hits.sort(key=lambda x: (-x["score"], x["docid"]))
    
    return flask.jsonify({"hits": hits})

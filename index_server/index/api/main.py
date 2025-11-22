"""REST API for Index Server."""
import re
import math
from pathlib import Path
import flask
import index

# Global data structures to hold the loaded index data
inverted_index = {}
pagerank = {}
stopwords = set()


def load_index():
    """Load inverted index, PageRank, and stopwords into memory."""
    global inverted_index, pagerank, stopwords
    
    # Load stopwords
    stopwords_path = Path(index.app.config["INDEX_PATH"]).parent.parent / "stopwords.txt"
    with open(stopwords_path, encoding="utf-8") as file:
        stopwords = set(line.strip() for line in file)
    
    # Load PageRank
    pagerank_path = Path(index.app.config["INDEX_PATH"]).parent.parent / "pagerank.out"
    with open(pagerank_path, encoding="utf-8") as file:
        for line in file:
            docid, score = line.strip().split(",")
            pagerank[int(docid)] = float(score)
    
    # Load inverted index
    index_path = Path(index.app.config["INDEX_PATH"])
    with open(index_path, encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            
            term = parts[0]
            idf = float(parts[1])
            
            # Parse document entries: docid, term_freq, normalization_factor
            doc_entries = []
            i = 2
            while i < len(parts):
                if i + 2 < len(parts):
                    docid = int(parts[i])
                    term_freq = int(parts[i + 1])
                    norm_factor = float(parts[i + 2])
                    doc_entries.append({
                        'docid': docid,
                        'term_freq': term_freq,
                        'norm_factor': norm_factor
                    })
                    i += 3
                else:
                    break
            
            inverted_index[term] = {
                'idf': idf,
                'docs': doc_entries
            }


def clean_query(query_string):
    """Clean query string by removing special chars and stopwords."""
    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9 ]+", "", query_string)
    text = text.casefold()
    
    # Split into terms
    terms = text.split()
    
    # Remove stopwords
    terms = [term for term in terms if term not in stopwords]
    
    return terms


def calculate_scores(query_terms, weight):
    """Calculate tf-idf and PageRank weighted scores for documents.
    
    Args:
        query_terms: List of cleaned query terms
        weight: Weight for PageRank (w), tf-idf weight is (1-w)
    
    Returns:
        List of dicts with docid and score, sorted by score descending
    """
    if not query_terms:
        return []
    
    # Find documents containing ALL query terms (AND query)
    # Start with documents from first term
    if query_terms[0] not in inverted_index:
        return []
    
    candidate_docs = set(doc['docid'] for doc in inverted_index[query_terms[0]]['docs'])
    
    # Intersect with documents from remaining terms
    for term in query_terms[1:]:
        if term not in inverted_index:
            return []  # If any term not in index, no results
        term_docs = set(doc['docid'] for doc in inverted_index[term]['docs'])
        candidate_docs = candidate_docs.intersection(term_docs)
    
    if not candidate_docs:
        return []
    
    # Calculate query vector
    query_vector = {}
    query_norm_factor = 0.0
    
    for term in query_terms:
        idf = inverted_index[term]['idf']
        # Term frequency in query (count occurrences)
        tf_query = query_terms.count(term)
        weight_unnormalized = tf_query * idf
        query_vector[term] = weight_unnormalized
        query_norm_factor += weight_unnormalized ** 2
    
    query_norm_factor = math.sqrt(query_norm_factor)
    
    # Normalize query vector
    for term in query_vector:
        query_vector[term] /= query_norm_factor
    
    # Calculate scores for each candidate document
    results = []
    for docid in candidate_docs:
        # Calculate tf-idf similarity (dot product of normalized vectors)
        tfidf_score = 0.0
        
        for term in query_terms:
            # Get document info for this term
            doc_info = None
            for doc in inverted_index[term]['docs']:
                if doc['docid'] == docid:
                    doc_info = doc
                    break
            
            if doc_info:
                idf = inverted_index[term]['idf']
                tf_doc = doc_info['term_freq']
                norm_factor = doc_info['norm_factor']
                
                # Normalized document term weight
                doc_term_weight = (tf_doc * idf) / norm_factor
                
                # Add to dot product
                tfidf_score += query_vector[term] * doc_term_weight
        
        # Get PageRank score (default to 0 if not found)
        pr_score = pagerank.get(docid, 0.0)
        
        # Calculate weighted score
        final_score = weight * pr_score + (1 - weight) * tfidf_score
        
        results.append({
            'docid': docid,
            'score': final_score
        })
    
    # Sort by score descending
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return results


@index.app.route('/api/v1/')
def get_api():
    """Return list of available services."""
    context = {
        "hits": "/api/v1/hits/",
        "url": "/api/v1/"
    }
    return flask.jsonify(context)


@index.app.route('/api/v1/hits/')
def get_hits():
    """Return search results for query."""
    # Get query parameter
    query = flask.request.args.get('q', '')
    
    # Get weight parameter (default 0.5)
    try:
        weight = float(flask.request.args.get('w', 0.5))
    except ValueError:
        weight = 0.5
    
    # Clean query
    query_terms = clean_query(query)
    
    # Calculate scores
    hits = calculate_scores(query_terms, weight)
    
    return flask.jsonify({"hits": hits})

import json
import re
import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# INSTRUCTION: Replace with your LDA NEG SageMaker endpoint name
# from the LDA Topic Modeling NEG notebook (STEP 6: Deploy Model)
ENDPOINT_NAME = "your-lda-neg-endpoint-name"

# CORS headers included in every response so the browser allows
# cross-origin requests from the web dashboard.
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST, OPTIONS"
}


# ============================================================
# VOCABULARY
# ============================================================
# INSTRUCTION: Replace this vocabulary list with the one from your
# LDA Topic Modeling NEG notebook. In your notebook at STEP 4
# (Vectorize the Text), run this command:
#
#     print(list(vectorizer.get_feature_names_out()))
#
# Then copy the entire output and paste it here, replacing the
# placeholder list below. The vocabulary must match exactly what
# your model was trained on, or topic predictions will be incorrect.
#
# The placeholder below contains ~500 common coffee review words
# so you can test the Lambda before deploying your own model.
# ============================================================

VOCABULARY = [
    "able", "about", "absolutely", "according", "actually", "after", "again",
    "ago", "all", "almost", "already", "also", "always", "amazon", "amount",
    "another", "any", "anything", "arrived", "auto", "away", "back", "bad",
    "barely", "basket", "batch", "beans", "bed", "been", "before", "best",
    "better", "big", "bit", "black", "bold", "bought", "brand", "brave",
    "break", "brew", "brewed", "brewer", "brewing", "broken", "brown",
    "burn", "burnt", "busy", "buy", "buying", "cafe", "came", "cap",
    "carafe", "care", "careful", "change", "cheap", "clean", "cleaning",
    "close", "coarse", "coffee", "coffeemaker", "cold", "come", "compact",
    "company", "complained", "complete", "consistent", "consumer",
    "convenient", "cost", "counter", "cream", "cup", "cups", "customer",
    "daily", "dark", "day", "days", "deal", "decided", "definitely",
    "delicious", "delivery", "design", "difference", "different",
    "disappointed", "done", "down", "drink", "drip", "due", "early",
    "ease", "easily", "easy", "electric", "else", "end", "enjoy", "enough",
    "espresso", "even", "ever", "every", "everyday", "everything",
    "excellent", "except", "expensive", "experience", "extra", "fact",
    "family", "fan", "far", "fast", "favorite", "feature", "features",
    "feel", "few", "fill", "filter", "filters", "finally", "find", "fine",
    "first", "fit", "flavor", "floor", "found", "free", "fresh", "friend",
    "front", "full", "getting", "give", "given", "glad", "glass", "goes",
    "gold", "gone", "good", "got", "great", "grind", "grinder", "grounds",
    "guess", "had", "half", "handle", "hands", "happy", "hard", "hate",
    "hear", "heat", "help", "high", "hold", "home", "hope", "hot", "hour",
    "hours", "house", "however", "husband", "ice", "into", "issue",
    "issues", "job", "just", "keep", "keeps", "kind", "kitchen", "knew",
    "large", "last", "lasted", "late", "later", "latte", "least", "leave",
    "left", "less", "let", "lid", "life", "light", "little", "long",
    "longer", "look", "looking", "lot", "love", "loved", "low", "machine",
    "machines", "main", "maker", "makers", "makes", "making", "market",
    "matter", "medium", "mess", "mind", "mine", "minutes", "model",
    "money", "month", "months", "morning", "mornings", "need", "needed",
    "never", "new", "next", "nice", "night", "noise", "none", "nothing",
    "number", "off", "office", "often", "old", "once", "ones", "open",
    "order", "ordered", "original", "otherwise", "out", "over", "overall",
    "package", "part", "parts", "past", "pay", "people", "perfect",
    "perfectly", "person", "pick", "piece", "place", "plastic", "pleased",
    "plus", "point", "poor", "pot", "pour", "power", "prefer", "pretty",
    "previous", "price", "problem", "problems", "product", "products",
    "program", "programmable", "pull", "pump", "purchased", "put",
    "quality", "quick", "quickly", "quiet", "quite", "ran", "rating",
    "ratings", "read", "real", "reason", "received", "recently",
    "recommend", "refund", "regular", "replace", "replaced", "replacement",
    "reported", "rest", "result", "return", "returned", "review", "reviews",
    "rich", "right", "room", "run", "running", "same", "satisfied", "scoop",
    "scoops", "seem", "seemed", "seems", "send", "sent", "serve",
    "service", "set", "several", "ship", "short", "shut", "side", "simple",
    "simply", "since", "single", "size", "sleep", "slow", "small", "smell",
    "smooth", "solid", "something", "sometimes", "soon", "sorry", "space",
    "special", "speed", "spend", "spill", "star", "stars", "start",
    "started", "stay", "steam", "still", "stop", "store", "stove",
    "strength", "strong", "stuff", "sugar", "suppose", "sure", "system",
    "takes", "taking", "tall", "taste", "tasted", "tastes", "tasting",
    "tea", "tell", "temperature", "terrible", "thank", "thanks", "then",
    "thought", "three", "time", "timer", "times", "tiny", "today",
    "together", "told", "too", "took", "top", "total", "tried", "trouble",
    "true", "truly", "try", "trying", "turn", "turned", "turns", "type",
    "under", "unit", "until", "upon", "use", "used", "using", "usually",
    "value", "wait", "wanted", "warm", "warranty", "waste", "water",
    "weak", "week", "weeks", "wife", "wish", "without", "woke",
    "wonderful", "work", "worked", "working", "works", "world", "worse",
    "worst", "worth", "wrong", "year", "years", "yes", "yet"
]


# Stopwords for text preprocessing (standard English stopwords)
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "don", "should", "now"
}


def preprocess_and_vectorize(text):
    """
    Preprocesses raw review text and converts it to a document-term vector.

    Steps:
        1. Lowercase the text
        2. Remove non-alphabetic characters (keep spaces)
        3. Tokenize (split on whitespace)
        4. Remove stopwords and single-character tokens
        5. Count occurrences of each vocabulary word

    Returns a list of integers (length = vocabulary size) where each value
    is the count of that vocabulary word in the review.
    """
    # Build a word-to-index lookup from the vocabulary
    vocab_index = {word: i for i, word in enumerate(VOCABULARY)}
    vocab_size = len(VOCABULARY)

    # Preprocess text
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    # Count vocabulary word occurrences
    vector = [0] * vocab_size
    for token in tokens:
        if token in vocab_index:
            vector[vocab_index[token]] += 1

    return vector


def lambda_handler(event, context):
    """
    AWS Lambda function for LDA topic modeling on NEGATIVE reviews.

    Receives raw negative review text from the web dashboard, preprocesses
    and vectorizes each review using the embedded vocabulary, then sends
    the document-term vectors to the LDA NEG SageMaker endpoint.

    Expected input:  { "instances": ["raw review text 1", "raw review text 2"] }
    Expected output: { "predictions": [{ "topic_mixture": [0.15, 0.25, ...] }, ...] }
    """
    logger.info("Received event: %s", event)

    # Handle CORS preflight request
    if event.get("httpMethod") == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": CORS_HEADERS,
            "body": ""
        }

    try:
        # Parse input (handle both direct invocation and API Gateway)
        if "body" in event:
            body = event["body"]
            if isinstance(body, str):
                body = json.loads(body)
        else:
            body = event

        instances = body.get("instances")

    except Exception as e:
        logger.error("Failed to parse input: %s", e)
        return {
            "statusCode": 400,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": "Invalid input format."})
        }

    if not instances:
        return {
            "statusCode": 400,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": "No instances provided."})
        }

    # Vectorize each review using the embedded vocabulary
    vectors = []
    for text in instances:
        vector = preprocess_and_vectorize(str(text))
        vectors.append(vector)

    logger.info("Vectorized %d reviews into %d-dim vectors", len(vectors), len(VOCABULARY))

    # Convert vectors to CSV format (one row per review, no header)
    csv_lines = []
    for vector in vectors:
        csv_lines.append(",".join(str(v) for v in vector))
    csv_payload = "\n".join(csv_lines)

    # Call the LDA NEG SageMaker endpoint
    try:
        runtime = boto3.client("sagemaker-runtime")
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Body=csv_payload
        )

        result = json.loads(response["Body"].read().decode("utf-8"))
        logger.info("Received topic distributions for %d reviews",
                     len(result.get("predictions", [])))

        return {
            "statusCode": 200,
            "headers": CORS_HEADERS,
            "body": json.dumps(result)
        }

    except Exception as e:
        logger.error("SageMaker invocation failed: %s", e)
        return {
            "statusCode": 500,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": f"Endpoint invocation failed: {str(e)}"})
        }

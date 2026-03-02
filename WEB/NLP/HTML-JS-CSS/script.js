// ============================================================
// Product Launch Command Center
// ============================================================
//
// This script powers the dashboard that analyzes customer feedback
// using three AWS SageMaker endpoints:
//
//   1. BlazingText  - Classifies reviews as Positive or Negative
//   2. LDA POS      - Discovers topics in positive reviews
//   3. LDA NEG      - Discovers topics in negative reviews
//
// The analysis runs in two phases:
//   Phase 1: BlazingText classifies all reviews (required)
//   Phase 2: Positive reviews go to LDA POS, negative to LDA NEG
//
// Students: You need to configure your API Gateway URLs in the
// Settings section at the bottom of the page before analyzing reviews.
// ============================================================


// ============================================================
// STOPWORDS
// ============================================================
// Common English words removed during preprocessing. These words
// appear frequently but carry little meaning for sentiment analysis.

const STOPWORDS = new Set([
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
    "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "can",
    "will", "just", "don", "should", "now", "ain", "aren", "couldn",
    "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn",
    "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won",
    "wouldn", "ll", "ve", "re", "also", "would", "could", "might",
    "shall", "may", "much", "many", "get", "got", "still", "really",
    "even", "well", "back", "one", "two", "go", "going", "went", "like",
    "make", "made", "come", "came", "take", "took", "know", "say",
    "said", "see", "seen", "thing", "things", "way", "want", "give"
]);


/**
 * Preprocesses a single review text for BlazingText classification.
 *
 * Steps:
 *   1. Convert to lowercase
 *   2. Remove URLs and HTML tags
 *   3. Remove non-alphabetic characters (keep spaces)
 *   4. Split into individual words (tokenize)
 *   5. Remove single-character words
 *   6. Remove common stopwords
 *   7. Rejoin into a single string
 *
 * NOTE: This does NOT include lemmatization (reducing words to root form,
 * e.g., "running" -> "run"). The SageMaker model was trained on lemmatized
 * text, so predictions on non-lemmatized input may be slightly less accurate.
 * This is a good discussion point about preprocessing consistency.
 */
function preprocessText(text) {
    // Step 1: Lowercase
    text = text.toLowerCase();

    // Step 2: Remove URLs and HTML tags
    text = text.replace(/https?:\/\/\S+|www\.\S+/g, "");
    text = text.replace(/<[^>]*>/g, "");

    // Step 3: Remove non-alphabetic characters (keep spaces)
    text = text.replace(/[^a-z\s]/g, "");

    // Steps 4-6: Tokenize, filter short words, remove stopwords
    var tokens = text.split(/\s+/)
        .filter(function(token) { return token.length > 1; })
        .filter(function(token) { return !STOPWORDS.has(token); });

    // Step 7: Rejoin
    return tokens.join(" ");
}


/**
 * Sends a POST request with JSON data to the given URL.
 * This generic function is used by all API calls (BlazingText, LDA POS, LDA NEG).
 */
async function postData(url, data) {
    var response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    });

    if (!response.ok) {
        throw new Error("HTTP error! Status: " + response.status);
    }

    return response.json();
}


/**
 * Sends reviews to the BlazingText endpoint for sentiment classification.
 * Reviews are preprocessed on the client side before sending.
 *
 * Returns an array of objects: { text, preprocessed, label, confidence }
 */
async function analyzeSentiment(reviews, apiUrl) {
    // Preprocess each review before sending
    var preprocessed = reviews.map(function(r) { return preprocessText(r); });

    // Send to BlazingText Lambda via API Gateway
    var payload = { instances: preprocessed };
    var response = await postData(apiUrl, payload);

    // Parse the response (handle API Gateway wrapper if present)
    var data = response;
    if (response.body && typeof response.body === "string") {
        data = JSON.parse(response.body);
    }

    // Map predictions back to original reviews
    var results = [];
    if (data.predictions && Array.isArray(data.predictions)) {
        data.predictions.forEach(function(pred, i) {
            // Handle nested array format from BlazingText
            var prediction = Array.isArray(pred) ? pred[0] : pred;

            // Extract and clean the label
            var rawLabel = prediction.label
                ? String(prediction.label).replace(/[\[\]]/g, "").trim()
                : "unknown";

            var label = rawLabel.toLowerCase().includes("positive")
                ? "POSITIVE"
                : rawLabel.toLowerCase().includes("negative")
                    ? "NEGATIVE"
                    : rawLabel.toUpperCase();

            // Extract confidence (probability)
            var confidence = prediction.prob
                ? parseFloat(String(prediction.prob).replace(/[\[\]]/g, ""))
                : 0;

            results.push({
                text: reviews[i] || "",
                preprocessed: preprocessed[i] || "",
                label: label,
                confidence: confidence
            });
        });
    }

    return results;
}


/**
 * Sends reviews to an LDA endpoint for topic discovery.
 * The Lambda function handles text vectorization (the vocabulary
 * is embedded in the Lambda), so we send raw review text here.
 *
 * Used for both LDA POS and LDA NEG endpoints.
 *
 * Returns: { topicDistributions: [[...], ...], averageDistribution: [...] }
 */
async function analyzeTopics(reviews, apiUrl) {
    // Send raw text (Lambda handles preprocessing and vectorization)
    var payload = { instances: reviews };
    var response = await postData(apiUrl, payload);

    // Parse the response
    var data = response;
    if (response.body && typeof response.body === "string") {
        data = JSON.parse(response.body);
    }

    // Extract topic distributions from each prediction
    var topicDistributions = [];
    if (data.predictions && Array.isArray(data.predictions)) {
        data.predictions.forEach(function(pred) {
            if (pred.topic_mixture) {
                topicDistributions.push(pred.topic_mixture);
            }
        });
    }

    // Calculate the average topic distribution across all reviews
    var numTopics = topicDistributions.length > 0
        ? topicDistributions[0].length
        : 0;

    var averageDistribution = [];
    for (var t = 0; t < numTopics; t++) {
        averageDistribution.push(0);
    }

    if (topicDistributions.length > 0) {
        topicDistributions.forEach(function(dist) {
            dist.forEach(function(weight, idx) {
                averageDistribution[idx] += weight;
            });
        });
        for (var t = 0; t < numTopics; t++) {
            averageDistribution[t] = averageDistribution[t] / topicDistributions.length;
        }
    }

    return {
        topicDistributions: topicDistributions,
        averageDistribution: averageDistribution
    };
}


/**
 * Escapes HTML special characters to prevent XSS when displaying user text.
 */
function escapeHtml(text) {
    var div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}


/**
 * Renders sentiment results as individual review cards with colored
 * badges (POSITIVE / NEGATIVE) and confidence bars.
 */
function renderSentimentResults(results) {
    var container = document.getElementById("sentimentResults");

    if (results.length === 0) {
        container.innerHTML = '<p class="no-results">No sentiment results to display.</p>';
        return;
    }

    var html = "";
    results.forEach(function(result, i) {
        var isPositive = result.label === "POSITIVE";
        var badgeClass = isPositive ? "badge-positive" : "badge-negative";
        var barClass = isPositive ? "confidence-fill-positive" : "confidence-fill-negative";
        var confidencePct = (result.confidence * 100).toFixed(1);

        // Truncate long reviews for display
        var displayText = result.text.length > 200
            ? result.text.substring(0, 200) + "..."
            : result.text;

        html += '<div class="review-card">'
            + '<div class="review-header">'
            + '<span class="review-number">Review ' + (i + 1) + '</span>'
            + '<span class="badge ' + badgeClass + '">' + result.label + '</span>'
            + '</div>'
            + '<p class="review-text">' + escapeHtml(displayText) + '</p>'
            + '<div class="confidence-row">'
            + '<span class="confidence-label">Confidence</span>'
            + '<div class="confidence-bar">'
            + '<div class="' + barClass + '" style="width: ' + confidencePct + '%"></div>'
            + '</div>'
            + '<span class="confidence-value">' + confidencePct + '%</span>'
            + '</div>'
            + '</div>';
    });

    container.innerHTML = html;
}


/**
 * Renders topic discovery results as a horizontal bar chart with
 * reviews grouped under their dominant (highest-weight) topic.
 *
 * @param {object} topicData - Object with averageDistribution and topicDistributions arrays
 * @param {string} containerId - DOM element ID to render into
 * @param {string[]} colorPalette - Array of hex colors for topic bars
 * @param {string[]} reviews - Original review texts corresponding to topicDistributions
 */
function renderTopicResults(topicData, containerId, colorPalette, reviews) {
    var container = document.getElementById(containerId);

    if (!topicData || topicData.averageDistribution.length === 0) {
        container.innerHTML = '<p class="no-results">No topic results to display.</p>';
        return;
    }

    var distribution = topicData.averageDistribution;
    var maxWeight = Math.max.apply(null, distribution);
    var numTopics = distribution.length;

    // Determine dominant topic for each review
    var reviewsByTopic = {};
    for (var t = 0; t < numTopics; t++) {
        reviewsByTopic[t] = [];
    }

    if (reviews && topicData.topicDistributions) {
        topicData.topicDistributions.forEach(function(dist, reviewIdx) {
            // Find the topic with the highest weight for this review
            var dominantTopic = 0;
            var maxTopicWeight = -1;
            dist.forEach(function(w, topicIdx) {
                if (w > maxTopicWeight) {
                    maxTopicWeight = w;
                    dominantTopic = topicIdx;
                }
            });

            var reviewText = reviews[reviewIdx] || "";
            var displayText = reviewText.length > 150
                ? reviewText.substring(0, 150) + "..."
                : reviewText;

            reviewsByTopic[dominantTopic].push({
                text: displayText,
                weight: maxTopicWeight
            });
        });
    }

    // Render bar chart with reviews grouped under each topic
    var html = '<div class="topic-chart">';
    distribution.forEach(function(weight, i) {
        var widthPct = maxWeight > 0 ? (weight / maxWeight) * 100 : 0;
        var color = colorPalette[i % colorPalette.length];
        var weightPct = (weight * 100).toFixed(1);
        var topicReviews = reviewsByTopic[i] || [];

        html += '<div class="topic-group">'
            + '<div class="topic-row">'
            + '<span class="topic-label">Topic ' + (i + 1) + '</span>'
            + '<div class="topic-bar-container">'
            + '<div class="topic-bar" style="width: ' + widthPct + '%; background-color: ' + color + ';"></div>'
            + '</div>'
            + '<span class="topic-weight">' + weightPct + '%</span>'
            + '<span class="topic-review-count">' + topicReviews.length + ' review' + (topicReviews.length !== 1 ? 's' : '') + '</span>'
            + '</div>';

        // Show reviews assigned to this topic
        if (topicReviews.length > 0) {
            html += '<div class="topic-reviews">';
            topicReviews.forEach(function(rev) {
                var topicPct = (rev.weight * 100).toFixed(0);
                html += '<div class="topic-review-item">'
                    + '<span class="topic-review-strength" style="color: ' + color + ';">' + topicPct + '%</span>'
                    + '<span class="topic-review-text">' + escapeHtml(rev.text) + '</span>'
                    + '</div>';
            });
            html += '</div>';
        }

        html += '</div>';
    });
    html += '</div>';

    container.innerHTML = html;
}


/**
 * Updates the five summary statistic cards with sentiment analysis results.
 * Includes: Total Reviews, Positive, Negative, Sentiment Score, Avg Confidence.
 */
function updateSummaryStats(results) {
    var statsContainer = document.getElementById("summaryStats");

    if (results.length === 0) {
        statsContainer.style.display = "none";
        return;
    }

    statsContainer.style.display = "flex";

    var total = results.length;
    var positive = results.filter(function(r) { return r.label === "POSITIVE"; }).length;
    var negative = results.filter(function(r) { return r.label === "NEGATIVE"; }).length;
    var sentimentScore = (positive / total) * 100;
    var sumConfidence = results.reduce(function(sum, r) { return sum + r.confidence; }, 0);
    var avgConfidence = sumConfidence / total;

    document.getElementById("statTotal").textContent = total;
    document.getElementById("statPositive").textContent = positive;
    document.getElementById("statNegative").textContent = negative;
    document.getElementById("statScore").textContent = sentimentScore.toFixed(0) + "%";
    document.getElementById("statConfidence").textContent = (avgConfidence * 100).toFixed(1) + "%";
}


/**
 * Controls the 2-step progress indicator.
 *
 * @param {number} step - 1 or 2
 * @param {string} state - "active", "completed", or "idle"
 */
function updateProgress(step, state) {
    var indicator = document.getElementById("progressIndicator");
    var step1 = document.getElementById("progressStep1");
    var step2 = document.getElementById("progressStep2");
    var connector = document.querySelector(".progress-connector");

    if (state === "hide") {
        indicator.style.display = "none";
        return;
    }

    indicator.style.display = "flex";

    if (step === 1 && state === "active") {
        step1.className = "progress-step active";
        step2.className = "progress-step";
        connector.className = "progress-connector";
    } else if (step === 1 && state === "completed") {
        step1.className = "progress-step completed";
        connector.className = "progress-connector filled";
    } else if (step === 2 && state === "active") {
        step1.className = "progress-step completed";
        step2.className = "progress-step active";
        connector.className = "progress-connector filled";
    } else if (step === 2 && state === "completed") {
        step1.className = "progress-step completed";
        step2.className = "progress-step completed";
        connector.className = "progress-connector filled";
    }
}


// Color palettes for topic bars
var POSITIVE_COLORS = [
    "#2E7D32", "#388E3C", "#43A047", "#4CAF50", "#66BB6A",
    "#81C784", "#A5D6A7", "#1B5E20", "#2E7D32", "#388E3C"
];

var NEGATIVE_COLORS = [
    "#C62828", "#D32F2F", "#E53935", "#F44336", "#EF5350",
    "#E57373", "#EF9A9A", "#B71C1C", "#C62828", "#D32F2F"
];


/**
 * Main handler: two-phase analysis orchestrator.
 *
 * Phase 1: BlazingText classifies all reviews as positive/negative
 * Phase 2: Positive reviews go to LDA POS, negative to LDA NEG (in parallel)
 *
 * BlazingText is required (LDA routing depends on it).
 * LDA endpoints are optional enhancements.
 */
async function analyzeFeedback() {
    var btn = document.getElementById("analyzeBtn");
    var inputText = document.getElementById("reviewInput").value.trim();

    // Parse reviews (one per line, skip empty lines)
    var reviews = inputText.split("\n")
        .map(function(line) { return line.trim(); })
        .filter(function(line) { return line.length > 0; });

    if (reviews.length === 0) {
        alert("Please enter at least one review to analyze.");
        return;
    }

    // Read API URLs from settings inputs
    var blazingTextUrl = document.getElementById("blazingTextUrl").value.trim();
    var ldaPosUrl = document.getElementById("ldaPosUrl").value.trim();
    var ldaNegUrl = document.getElementById("ldaNegUrl").value.trim();

    if (!blazingTextUrl) {
        alert("Please configure the BlazingText API Gateway URL in the Settings section below. Sentiment classification is required before topic analysis can run.");
        return;
    }

    // Update button to loading state
    btn.disabled = true;
    btn.textContent = "Analyzing...";

    // Show results area and reset panels
    document.getElementById("resultsArea").style.display = "block";
    document.getElementById("sentimentPanel").style.display = "";
    document.getElementById("sentimentResults").innerHTML = '<p class="loading">Classifying sentiment...</p>';
    document.getElementById("topicPosResults").innerHTML = "";
    document.getElementById("topicNegResults").innerHTML = "";
    document.getElementById("summaryStats").style.display = "none";

    // Hide topic panels initially (show them only if LDA URLs are configured)
    document.getElementById("topicPosPanel").style.display = ldaPosUrl ? "" : "none";
    document.getElementById("topicNegPanel").style.display = ldaNegUrl ? "" : "none";

    // ---- PHASE 1: Sentiment Classification ----
    updateProgress(1, "active");

    var sentimentResults = null;
    try {
        sentimentResults = await analyzeSentiment(reviews, blazingTextUrl);
        renderSentimentResults(sentimentResults);
        updateSummaryStats(sentimentResults);
        updateProgress(1, "completed");
    } catch (error) {
        document.getElementById("sentimentResults").innerHTML =
            '<p class="error-message">Sentiment analysis failed: '
            + escapeHtml(error.message) + '</p>';
        updateProgress("hide");
        btn.disabled = false;
        btn.textContent = "Analyze Feedback";
        return;
    }

    // ---- PHASE 2: Topic Discovery ----
    if (!ldaPosUrl && !ldaNegUrl) {
        // No LDA endpoints configured, we are done
        updateProgress(2, "completed");
        btn.disabled = false;
        btn.textContent = "Analyze Feedback";
        return;
    }

    updateProgress(2, "active");

    // Split reviews by sentiment label
    var positiveReviews = [];
    var negativeReviews = [];
    sentimentResults.forEach(function(result) {
        if (result.label === "POSITIVE") {
            positiveReviews.push(result.text);
        } else {
            negativeReviews.push(result.text);
        }
    });

    // Show loading states for configured LDA panels
    if (ldaPosUrl) {
        document.getElementById("topicPosPanel").style.display = "";
        if (positiveReviews.length > 0) {
            document.getElementById("topicPosResults").innerHTML =
                '<p class="loading">Discovering positive topics...</p>';
        } else {
            document.getElementById("topicPosResults").innerHTML =
                '<p class="info-message">No positive reviews found. All reviews were classified as negative.</p>';
        }
    }

    if (ldaNegUrl) {
        document.getElementById("topicNegPanel").style.display = "";
        if (negativeReviews.length > 0) {
            document.getElementById("topicNegResults").innerHTML =
                '<p class="loading">Discovering negative topics...</p>';
        } else {
            document.getElementById("topicNegResults").innerHTML =
                '<p class="info-message">No negative reviews found. All reviews were classified as positive.</p>';
        }
    }

    // Build LDA promises (only for configured endpoints with reviews)
    var ldaPromises = [];
    var ldaLabels = [];
    var ldaReviewSets = [];

    if (ldaPosUrl && positiveReviews.length > 0) {
        ldaPromises.push(analyzeTopics(positiveReviews, ldaPosUrl));
        ldaLabels.push("pos");
        ldaReviewSets.push(positiveReviews);
    }

    if (ldaNegUrl && negativeReviews.length > 0) {
        ldaPromises.push(analyzeTopics(negativeReviews, ldaNegUrl));
        ldaLabels.push("neg");
        ldaReviewSets.push(negativeReviews);
    }

    if (ldaPromises.length > 0) {
        try {
            var ldaResults = await Promise.allSettled(ldaPromises);

            ldaResults.forEach(function(result, i) {
                var label = ldaLabels[i];
                var reviewSet = ldaReviewSets[i];

                if (result.status === "fulfilled") {
                    if (label === "pos") {
                        renderTopicResults(result.value, "topicPosResults", POSITIVE_COLORS, reviewSet);
                    } else if (label === "neg") {
                        renderTopicResults(result.value, "topicNegResults", NEGATIVE_COLORS, reviewSet);
                    }
                } else {
                    if (label === "pos") {
                        document.getElementById("topicPosResults").innerHTML =
                            '<p class="error-message">Positive topic analysis failed: '
                            + escapeHtml(result.reason.message) + '</p>';
                    } else if (label === "neg") {
                        document.getElementById("topicNegResults").innerHTML =
                            '<p class="error-message">Negative topic analysis failed: '
                            + escapeHtml(result.reason.message) + '</p>';
                    }
                }
            });
        } catch (error) {
            // Unexpected error in Promise.allSettled (should not happen)
            if (ldaPosUrl) {
                document.getElementById("topicPosResults").innerHTML =
                    '<p class="error-message">Topic analysis error: '
                    + escapeHtml(error.message) + '</p>';
            }
        }
    }

    updateProgress(2, "completed");
    btn.disabled = false;
    btn.textContent = "Analyze Feedback";
}


/**
 * Clears all results and resets the display to its initial state.
 */
function clearResults() {
    document.getElementById("resultsArea").style.display = "none";
    document.getElementById("sentimentResults").innerHTML = "";
    document.getElementById("topicPosResults").innerHTML = "";
    document.getElementById("topicNegResults").innerHTML = "";
    document.getElementById("summaryStats").style.display = "none";
    updateProgress(0, "hide");

    // Show all panels again (they may have been hidden)
    document.getElementById("sentimentPanel").style.display = "";
    document.getElementById("topicPosPanel").style.display = "";
    document.getElementById("topicNegPanel").style.display = "";
}


/**
 * Loads saved API Gateway URLs from localStorage into the settings inputs.
 */
function loadSettings() {
    var blazingTextUrl = localStorage.getItem("productLaunch_blazingTextUrl") || "";
    var ldaPosUrl = localStorage.getItem("productLaunch_ldaPosUrl") || "";
    var ldaNegUrl = localStorage.getItem("productLaunch_ldaNegUrl") || "";

    document.getElementById("blazingTextUrl").value = blazingTextUrl;
    document.getElementById("ldaPosUrl").value = ldaPosUrl;
    document.getElementById("ldaNegUrl").value = ldaNegUrl;
}


/**
 * Saves API Gateway URLs from the settings inputs to localStorage
 * so they persist across page reloads.
 */
function saveSettings() {
    var blazingTextUrl = document.getElementById("blazingTextUrl").value.trim();
    var ldaPosUrl = document.getElementById("ldaPosUrl").value.trim();
    var ldaNegUrl = document.getElementById("ldaNegUrl").value.trim();

    localStorage.setItem("productLaunch_blazingTextUrl", blazingTextUrl);
    localStorage.setItem("productLaunch_ldaPosUrl", ldaPosUrl);
    localStorage.setItem("productLaunch_ldaNegUrl", ldaNegUrl);

    // Show brief confirmation on the button
    var btn = document.getElementById("saveSettingsBtn");
    var originalText = btn.textContent;
    btn.textContent = "Saved!";
    btn.classList.add("btn-saved");
    setTimeout(function() {
        btn.textContent = originalText;
        btn.classList.remove("btn-saved");
    }, 2000);
}


// ============================================================
// Initialize on page load
// ============================================================
document.addEventListener("DOMContentLoaded", function() {
    loadSettings();

    document.getElementById("analyzeBtn").addEventListener("click", analyzeFeedback);
    document.getElementById("clearBtn").addEventListener("click", clearResults);
    document.getElementById("saveSettingsBtn").addEventListener("click", saveSettings);
});

// ============================================================
// Ames Housing Market Segmentation Tool
// ============================================================
//
// This script powers the dashboard that classifies properties into
// market tiers using an AWS SageMaker KMeans clustering endpoint.
//
// The flow:
//   1. User enters 5 key property features (or loads an example)
//   2. Features are sent as JSON to API Gateway
//   3. Lambda builds a full 74-feature vector with median defaults,
//      applies log-transform, standardization, and PCA reduction
//   4. Lambda sends 2 PCA values to the KMeans endpoint
//   5. Cluster assignment and tier profile are displayed
//
// Students: Configure your API Gateway URL in the Settings section
// at the bottom of the page before classifying properties.
// ============================================================


// ============================================================
// CLUSTER PROFILES
// ============================================================
// INSTRUCTION: Configure one entry per cluster. The number of clusters (K)
// depends on your model — your notebook's silhouette analysis determines the
// best K. Add or remove entries below to match your K value.
//
// In your notebook's Visualize Clusters step, look at the
// "Cluster Profile Summary" table that shows the average values
// for each cluster. Use those values to fill in the profiles below.
//
// IMPORTANT: The cluster numbers (0, 1, 2, ...) may map to different
// tiers depending on your training run. Check which cluster has
// the highest average SalePrice (that is your higher-value tier),
// which has the lowest (lower-value tier), and so on. Update the
// mapping below accordingly.
//
// If your model uses K=3, add a third entry for cluster "2".
// ============================================================

var CLUSTER_PROFILES = {
    // INSTRUCTION: Update these profiles from your notebook output.
    //
    // In your notebook's Visualize Clusters step, look at the
    // "Cluster Profile Summary" table printed below the plots. It shows
    // each cluster's average SalePrice, Quality, Living Area, etc.
    //
    // 1. Check how many clusters your model uses (K value from the
    //    silhouette analysis). Add one entry per cluster below.
    // 2. Identify which cluster number maps to which tier by looking
    //    at the average SalePrice column.
    // 3. Replace the placeholder values below with your actual averages.
    // 4. If your model uses K=3, add a third entry for cluster "2".
    //
    // The values below are PLACEHOLDERS for K=2. Your model's actual
    // averages will differ. If you skip this step, the dashboard will
    // still work but will show incorrect profile statistics.

    "0": {
        "name": "Higher-Value Tier",
        "description": "Properties with higher quality ratings, larger living areas, and newer construction. The premium segment of the Ames market.",
        "color": "var(--tier-premium, #B8860B)",
        "avgPrice": "$230,000",
        "avgQuality": "7",
        "avgLivingArea": "1,800 sq ft",
        "avgYearBuilt": "1995",
        "avgBasement": "1,100 sq ft",
        "avgGarage": "550 sq ft"
    },
    "1": {
        "name": "Lower-Value Tier",
        "description": "Properties with lower overall quality, smaller living areas, and older construction. Typically more affordable homes in the Ames market.",
        "color": "var(--tier-budget, #4C72B0)",
        "avgPrice": "$130,000",
        "avgQuality": "5",
        "avgLivingArea": "1,100 sq ft",
        "avgYearBuilt": "1960",
        "avgBasement": "800 sq ft",
        "avgGarage": "380 sq ft"
    }

    // INSTRUCTION: If your model uses K=3, uncomment and configure this entry:
    // "2": {
    //     "name": "Premium Tier",
    //     "description": "High-quality, spacious homes with premium finishes and modern construction.",
    //     "color": "var(--tier-midrange, #0D9488)",
    //     "avgPrice": "$280,000",
    //     "avgQuality": "8",
    //     "avgLivingArea": "2,200 sq ft",
    //     "avgYearBuilt": "2003",
    //     "avgBasement": "1,200 sq ft",
    //     "avgGarage": "650 sq ft"
    // }
};


// ============================================================
// EXAMPLE PROPERTIES
// ============================================================
// Three sample properties spanning the market range from low to high value.
// These examples work regardless of how many clusters (K) your model uses.
// With K=2, the budget and mid-range examples may land in the same cluster.

var EXAMPLES = {
    "budget": {
        "overallQual": 4,
        "grLivArea": 864,
        "yearBuilt": 1950,
        "totalBsmtSF": 600,
        "garageArea": 280
    },
    "midrange": {
        "overallQual": 6,
        "grLivArea": 1500,
        "yearBuilt": 1995,
        "totalBsmtSF": 1000,
        "garageArea": 480
    },
    "premium": {
        "overallQual": 9,
        "grLivArea": 2400,
        "yearBuilt": 2008,
        "totalBsmtSF": 1400,
        "garageArea": 700
    }
};


/**
 * Collects the 5 input values from the form fields and validates them.
 *
 * @returns {object} - JSON object: { instances: [[qual, area, year, bsmt, garage]] }
 * @throws {Error} - If any field is empty or out of range.
 */
function collectInputs() {
    var fields = [
        { id: "overallQual", label: "Overall Quality", min: 1, max: 10 },
        { id: "grLivArea", label: "Above-Ground Living Area", min: 300, max: 6000 },
        { id: "yearBuilt", label: "Year Built", min: 1870, max: 2025 },
        { id: "totalBsmtSF", label: "Total Basement Area", min: 0, max: 6000 },
        { id: "garageArea", label: "Garage Area", min: 0, max: 1500 }
    ];

    var values = [];

    for (var i = 0; i < fields.length; i++) {
        var field = fields[i];
        var input = document.getElementById(field.id);
        var raw = input.value.trim();

        if (!raw) {
            throw new Error("Please enter a value for " + field.label + ".");
        }

        var num = parseFloat(raw);

        if (isNaN(num)) {
            throw new Error(field.label + " must be a number.");
        }

        if (num < field.min || num > field.max) {
            throw new Error(
                field.label + " must be between " + field.min + " and " + field.max + "."
            );
        }

        values.push(num);
    }

    return { instances: [values] };
}


/**
 * Sends a POST request with JSON data to the given URL.
 *
 * @param {string} url - The API Gateway endpoint URL.
 * @param {object} data - The JSON payload to send.
 * @returns {Promise<object>} - The parsed JSON response.
 */
async function postData(url, data) {
    var controller = new AbortController();
    var timeoutId = setTimeout(function() { controller.abort(); }, 30000);

    try {
        var response = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
            signal: controller.signal
        });

        if (!response.ok) {
            throw new Error("HTTP error! Status: " + response.status);
        }

        return response.json();
    } finally {
        clearTimeout(timeoutId);
    }
}


/**
 * Escapes HTML special characters to prevent XSS when displaying text.
 */
function escapeHtml(text) {
    var div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}


/**
 * Renders the cluster classification result in the results panel.
 *
 * @param {object} clusterData - Object with closest_cluster and distance_to_cluster.
 */
function renderResult(clusterData) {
    var container = document.getElementById("classificationResult");
    var cluster = String(clusterData.closest_cluster);
    var distance = clusterData.distance_to_cluster;

    // Look up the cluster profile
    var profile = CLUSTER_PROFILES[cluster];

    if (!profile) {
        container.innerHTML =
            '<div class="tier-display">'
            + '  <div class="tier-badge" style="background-color: #666;">Cluster ' + escapeHtml(cluster) + '</div>'
            + '  <div class="tier-description">Cluster ' + escapeHtml(cluster) + ' is not configured. '
            + 'Your model returned a cluster that does not have a matching entry in CLUSTER_PROFILES. '
            + 'Add an entry for cluster "' + escapeHtml(cluster) + '" in script.js based on your '
            + 'notebook\'s Visualize Clusters output.</div>'
            + '</div>';
        return;
    }

    var html = ''
        // Tier badge and description
        + '<div class="tier-display">'
        + '  <div class="tier-badge" style="background-color: ' + profile.color + ';">'
        + escapeHtml(profile.name)
        + '  </div>'
        + '  <div class="tier-description">' + escapeHtml(profile.description) + '</div>'
        + '</div>'

        // Cluster profile stats
        + '<div class="profile-stats">'
        + '  <h3 class="stats-title">Cluster Profile (Average Values)</h3>'
        + '  <div class="stats-grid">'
        + '    <div class="stat-item">'
        + '      <span class="stat-label">Avg Sale Price</span>'
        + '      <span class="stat-value price-stat">' + escapeHtml(profile.avgPrice) + '</span>'
        + '    </div>'
        + '    <div class="stat-item">'
        + '      <span class="stat-label">Avg Quality</span>'
        + '      <span class="stat-value">' + escapeHtml(profile.avgQuality) + ' / 10</span>'
        + '    </div>'
        + '    <div class="stat-item">'
        + '      <span class="stat-label">Avg Living Area</span>'
        + '      <span class="stat-value">' + escapeHtml(profile.avgLivingArea) + '</span>'
        + '    </div>'
        + '    <div class="stat-item">'
        + '      <span class="stat-label">Avg Year Built</span>'
        + '      <span class="stat-value">' + escapeHtml(profile.avgYearBuilt) + '</span>'
        + '    </div>'
        + '    <div class="stat-item">'
        + '      <span class="stat-label">Avg Basement</span>'
        + '      <span class="stat-value">' + escapeHtml(profile.avgBasement) + '</span>'
        + '    </div>'
        + '    <div class="stat-item">'
        + '      <span class="stat-label">Avg Garage</span>'
        + '      <span class="stat-value">' + escapeHtml(profile.avgGarage) + '</span>'
        + '    </div>'
        + '  </div>'
        + '</div>'

        // Technical details
        + '<div class="result-details">'
        + '  <div class="detail-item">'
        + '    <span class="detail-label">Cluster Number</span>'
        + '    <span class="detail-value">' + cluster + '</span>'
        + '  </div>'
        + '  <div class="detail-item">'
        + '    <span class="detail-label">Distance to Centroid</span>'
        + '    <span class="detail-value">' + distance.toFixed(4) + '</span>'
        + '    <span class="detail-hint">Lower = more typical of this tier</span>'
        + '  </div>'
        + '  <div class="detail-item">'
        + '    <span class="detail-label">Model</span>'
        + '    <span class="detail-value">KMeans (PCA-reduced)</span>'
        + '  </div>'
        + '</div>';

    container.innerHTML = html;
}


/**
 * Controls the 2-step progress indicator.
 *
 * @param {number} step - 1 or 2
 * @param {string} state - "active", "completed", or "hide"
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


/**
 * Main handler: sends property features to the classification endpoint
 * and displays the cluster result.
 */
async function predictSegment() {
    var btn = document.getElementById("classifyBtn");

    // Read API URL from settings
    var classifyUrl = document.getElementById("classifyUrl").value.trim();

    if (!classifyUrl) {
        alert("Please configure the Classification API URL in the Settings section below.");
        return;
    }

    // Collect and validate inputs
    var payload;
    try {
        payload = collectInputs();
    } catch (error) {
        document.getElementById("resultsArea").style.display = "block";
        document.getElementById("classificationResult").innerHTML =
            '<p class="error-message">' + escapeHtml(error.message) + '</p>';
        return;
    }

    // Update button to loading state
    btn.disabled = true;
    btn.textContent = "Classifying...";

    // Show results area and progress
    document.getElementById("resultsArea").style.display = "block";
    document.getElementById("classificationResult").innerHTML =
        '<p class="loading">Sending features to SageMaker endpoint...</p>';

    // Phase 1: Sending features
    updateProgress(1, "active");

    try {
        // Phase 2: Classifying property
        updateProgress(1, "completed");
        updateProgress(2, "active");

        var response = await postData(classifyUrl, payload);

        // Parse the response (handle API Gateway wrapper if present)
        var data = response;
        if (response.body && typeof response.body === "string") {
            data = JSON.parse(response.body);
        }

        // Extract the cluster prediction from the response
        var prediction = null;

        if (data.predictions) {
            var predictions = data.predictions;

            // Handle nested structure
            if (predictions.predictions && Array.isArray(predictions.predictions)) {
                predictions = predictions.predictions;
            }

            if (Array.isArray(predictions) && predictions.length > 0) {
                prediction = predictions[0];
            }
        }

        if (!prediction || prediction.closest_cluster === undefined) {
            throw new Error(
                "Could not extract a cluster assignment from the response. Raw response: "
                + JSON.stringify(response).substring(0, 200)
            );
        }

        renderResult(prediction);
        updateProgress(2, "completed");

    } catch (error) {
        var errorMsg = error.message;
        if (error.name === "AbortError") {
            errorMsg = "Request timed out after 30 seconds. Common causes: "
                + "(1) The SageMaker endpoint may still be starting up. "
                + "(2) The Lambda function may have timed out — check CloudWatch Logs. "
                + "(3) The API URL in Settings may be incorrect.";
        } else if (errorMsg === "Failed to fetch") {
            errorMsg = "Could not reach the API. Common causes: "
                + "(1) The API URL in Settings is incorrect or missing the /classify path. "
                + "(2) The API Gateway has not been deployed yet. "
                + "(3) CORS is not configured — ensure the API Gateway has an OPTIONS method on /classify. "
                + "Open the browser console (F12) for more details.";
        }
        document.getElementById("classificationResult").innerHTML =
            '<p class="error-message">Classification failed: ' + escapeHtml(errorMsg) + '</p>';
        updateProgress(0, "hide");
    } finally {
        btn.disabled = false;
        btn.textContent = "Classify Property";
    }
}


/**
 * Loads example property data into the form fields.
 *
 * @param {string} exampleId - The example key ("budget", "midrange", or "premium" — these describe input properties, not cluster assignments).
 */
function loadExample(exampleId) {
    var data = EXAMPLES[exampleId];
    if (data) {
        document.getElementById("overallQual").value = data.overallQual;
        document.getElementById("grLivArea").value = data.grLivArea;
        document.getElementById("yearBuilt").value = data.yearBuilt;
        document.getElementById("totalBsmtSF").value = data.totalBsmtSF;
        document.getElementById("garageArea").value = data.garageArea;
    }
}


/**
 * Clears the results and resets the display.
 */
function clearResults() {
    document.getElementById("overallQual").value = "";
    document.getElementById("grLivArea").value = "";
    document.getElementById("yearBuilt").value = "";
    document.getElementById("totalBsmtSF").value = "";
    document.getElementById("garageArea").value = "";
    document.getElementById("resultsArea").style.display = "none";
    document.getElementById("classificationResult").innerHTML = "";
    updateProgress(0, "hide");
}


/**
 * Loads saved API Gateway URL from localStorage into the settings input.
 */
function loadSettings() {
    var classifyUrl = localStorage.getItem("pcaKmeans_classifyUrl") || "";
    document.getElementById("classifyUrl").value = classifyUrl;
}


/**
 * Saves the API Gateway URL from the settings input to localStorage
 * so it persists across page reloads.
 */
function saveSettings() {
    var classifyUrl = document.getElementById("classifyUrl").value.trim();
    localStorage.setItem("pcaKmeans_classifyUrl", classifyUrl);

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

    document.getElementById("classifyBtn").addEventListener("click", predictSegment);
    document.getElementById("clearBtn").addEventListener("click", clearResults);
    document.getElementById("saveSettingsBtn").addEventListener("click", saveSettings);

    // Submit form on Enter key from any input field
    var inputFields = document.querySelectorAll(".form-grid input[type='number']");
    inputFields.forEach(function(input) {
        input.addEventListener("keydown", function(e) {
            if (e.key === "Enter") {
                e.preventDefault();
                predictSegment();
            }
        });
    });

    // Attach example button handlers
    var exampleButtons = document.querySelectorAll(".btn-example");
    exampleButtons.forEach(function(btn) {
        btn.addEventListener("click", function() {
            loadExample(this.getAttribute("data-example"));
        });
    });
});
// ============================================================
// Ames Housing Price Predictor
// ============================================================
//
// This script powers the dashboard that predicts house sale prices
// using an AWS SageMaker Linear Learner regression endpoint.
//
// The flow:
//   1. User pastes 203 comma-separated feature values (or loads an example)
//   2. Features are parsed and sent as JSON to API Gateway
//   3. Lambda converts to CSV and forwards to SageMaker endpoint
//   4. Predicted sale price is displayed
//
// Students: Configure your API Gateway URL in the Settings section
// at the bottom of the page before making predictions.
// ============================================================


// ============================================================
// EXAMPLE DATA
// ============================================================
// Three sample houses from the Ames Housing validation set.
// Each array contains 203 numeric feature values.

var EXAMPLES = {
    "1": [
        1048, 527453160, 160, 24, 2308, 6, 6, 1975, 1975, 0,
        286, 294, 275, 855, 855, 601, 0, 1456, 0, 0,
        2, 1, 4, 1, 7, 0, 1975, 2, 460, 0,
        0, 0, 0, 0, 0, 0, 5, 2008, 0.0078632479, 0.075555556,
        0.0075213675, 0, 0, 0, 0, 1, 0, 1, 0, 0,
        1, 0, 0, 1, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 1, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
        0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 1, 1, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 1, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 0
    ],
    "2": [
        2522, 535376150, 81, 24, 1944, 5, 6, 1953, 1953, 0,
        0, 380, 0, 630, 630, 682, 0, 1312, 0, 0,
        1, 0, 3, 1, 5, 1, 1953, 1, 264, 0,
        0, 0, 0, 0, 0, 0, 7, 2006, 0.0049019608, 0.044444444,
        0.0041666667, 0, 0, 0, 0, 1, 0, 1, 0, 0,
        1, 0, 0, 0, 1, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 1, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
        0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
        0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 1, 1, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 1, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 0
    ],
    "3": [
        1188, 527162130, 152, 24, 3608, 7, 5, 1978, 1978, 0,
        108, 468, 180, 756, 756, 860, 0, 1616, 1, 0,
        2, 0, 3, 1, 7, 1, 1978, 2, 440, 0,
        0, 0, 0, 0, 0, 0, 5, 2007, 0.0060975610, 0.066666667,
        0.0058823529, 0, 0, 0, 0, 1, 0, 1, 0, 0,
        1, 0, 0, 1, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 1, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
        0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 1, 1, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 1, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 1, 0
    ]
};


/**
 * Parses a comma-separated string of numbers into the JSON format
 * expected by the Lambda function.
 *
 * @param {string} raw - Comma-separated numeric values from the textarea.
 * @returns {object} - JSON object: { instances: [[number, number, ...]] }
 */
function parseFeatureInput(raw) {
    var numbers = raw.split(",").map(function(val) {
        var num = parseFloat(val.trim());
        if (isNaN(num)) {
            throw new Error("Invalid number encountered: " + val.trim());
        }
        return num;
    });

    if (numbers.length !== 203) {
        throw new Error(
            "Expected 203 features but received " + numbers.length + ". "
            + "Make sure you are pasting a complete row from your validation set."
        );
    }

    return { instances: [numbers] };
}


/**
 * Sends a POST request with JSON data to the given URL.
 *
 * @param {string} url - The API Gateway endpoint URL.
 * @param {object} data - The JSON payload to send.
 * @returns {Promise<object>} - The parsed JSON response.
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
 * Escapes HTML special characters to prevent XSS when displaying text.
 */
function escapeHtml(text) {
    var div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}


/**
 * Formats a number as US dollars (e.g., "$178,532").
 *
 * @param {number} value - The numeric price to format.
 * @returns {string} - Formatted dollar string.
 */
function formatPrice(value) {
    return "$" + Math.round(value).toLocaleString("en-US");
}


/**
 * Renders the prediction result in the results panel.
 *
 * @param {number} price - The predicted sale price from SageMaker.
 * @param {number} featureCount - The number of features sent.
 */
function renderResult(price, featureCount) {
    var container = document.getElementById("predictionResult");

    var html = '<div class="price-display">'
        + '<div class="price-label">Predicted Sale Price</div>'
        + '<div class="price-value">' + formatPrice(price) + '</div>'
        + '</div>'
        + '<div class="result-details">'
        + '<div class="detail-item">'
        + '<span class="detail-label">Features Sent</span>'
        + '<span class="detail-value">' + featureCount + '</span>'
        + '</div>'
        + '<div class="detail-item">'
        + '<span class="detail-label">Model</span>'
        + '<span class="detail-value">Linear Learner (Regression)</span>'
        + '</div>'
        + '<div class="detail-item">'
        + '<span class="detail-label">Raw Score</span>'
        + '<span class="detail-value">' + price.toFixed(2) + '</span>'
        + '</div>'
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
 * Main handler: sends feature data to the prediction endpoint
 * and displays the result.
 */
async function predictPrice() {
    var btn = document.getElementById("predictBtn");
    var inputText = document.getElementById("featureInput").value.trim();

    if (!inputText) {
        alert("Please enter feature values or click an Example button to load sample data.");
        return;
    }

    // Read API URL from settings
    var predictUrl = document.getElementById("predictUrl").value.trim();

    if (!predictUrl) {
        alert("Please configure the Prediction API URL in the Settings section below.");
        return;
    }

    // Parse input
    var payload;
    try {
        payload = parseFeatureInput(inputText);
    } catch (error) {
        document.getElementById("resultsArea").style.display = "block";
        document.getElementById("predictionResult").innerHTML =
            '<p class="error-message">' + escapeHtml(error.message) + '</p>';
        return;
    }

    // Update button to loading state
    btn.disabled = true;
    btn.textContent = "Predicting...";

    // Show results area and progress
    document.getElementById("resultsArea").style.display = "block";
    document.getElementById("predictionResult").innerHTML =
        '<p class="loading">Sending features to SageMaker endpoint...</p>';

    // Phase 1: Sending features
    updateProgress(1, "active");

    try {
        // Phase 2: Getting prediction
        updateProgress(1, "completed");
        updateProgress(2, "active");

        var response = await postData(predictUrl, payload);

        // Parse the response (handle API Gateway wrapper if present)
        var data = response;
        if (response.body && typeof response.body === "string") {
            data = JSON.parse(response.body);
        }

        // Extract the predicted price from the response
        var price = null;

        if (data.predictions) {
            var predictions = data.predictions;

            // Handle nested structure: { predictions: { predictions: [{ score: N }] } }
            if (predictions.predictions && Array.isArray(predictions.predictions)) {
                predictions = predictions.predictions;
            }

            if (Array.isArray(predictions) && predictions.length > 0) {
                var firstPrediction = predictions[0];
                if (typeof firstPrediction === "object" && firstPrediction.score !== undefined) {
                    price = parseFloat(firstPrediction.score);
                } else if (typeof firstPrediction === "number") {
                    price = firstPrediction;
                }
            }
        }

        if (price === null || isNaN(price)) {
            throw new Error("Could not extract a prediction from the response. Raw response: "
                + JSON.stringify(response).substring(0, 200));
        }

        renderResult(price, payload.instances[0].length);
        updateProgress(2, "completed");

    } catch (error) {
        document.getElementById("predictionResult").innerHTML =
            '<p class="error-message">Prediction failed: ' + escapeHtml(error.message) + '</p>';
        updateProgress(0, "hide");
    } finally {
        btn.disabled = false;
        btn.textContent = "Predict Price";
    }
}


/**
 * Loads example feature data into the textarea.
 *
 * @param {string} exampleId - The example key ("1", "2", or "3").
 */
function loadExample(exampleId) {
    var data = EXAMPLES[exampleId];
    if (data) {
        document.getElementById("featureInput").value = data.join(", ");
    }
}


/**
 * Clears the results and resets the display.
 */
function clearResults() {
    document.getElementById("featureInput").value = "";
    document.getElementById("resultsArea").style.display = "none";
    document.getElementById("predictionResult").innerHTML = "";
    updateProgress(0, "hide");
}


/**
 * Loads saved API Gateway URL from localStorage into the settings input.
 */
function loadSettings() {
    var predictUrl = localStorage.getItem("amesHousing_predictUrl") || "";
    document.getElementById("predictUrl").value = predictUrl;
}


/**
 * Saves the API Gateway URL from the settings input to localStorage
 * so it persists across page reloads.
 */
function saveSettings() {
    var predictUrl = document.getElementById("predictUrl").value.trim();
    localStorage.setItem("amesHousing_predictUrl", predictUrl);

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

    document.getElementById("predictBtn").addEventListener("click", predictPrice);
    document.getElementById("clearBtn").addEventListener("click", clearResults);
    document.getElementById("saveSettingsBtn").addEventListener("click", saveSettings);

    // Attach example button handlers
    var exampleButtons = document.querySelectorAll(".btn-example");
    exampleButtons.forEach(function(btn) {
        btn.addEventListener("click", function() {
            loadExample(this.getAttribute("data-example"));
        });
    });
});

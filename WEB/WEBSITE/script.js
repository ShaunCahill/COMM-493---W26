// ==========================================
// TODO: PASTE YOUR API GATEWAY URL HERE
// Example: "https://xyz123.execute-api.us-east-1.amazonaws.com/dev/predict"
// ==========================================
const API_URL = "YOUR_API_GATEWAY_URL_HERE"; 

async function makePrediction() {
    const inputField = document.getElementById("dataInput");
    const resultContainer = document.getElementById("resultContainer");
    const predictionOutput = document.getElementById("predictionOutput");
    const errorContainer = document.getElementById("errorContainer");
    const errorMessage = document.getElementById("errorMessage");

    // Reset UI
    resultContainer.classList.add("hidden");
    errorContainer.classList.add("hidden");
    
    // 1. Get and Clean the Input Data
    const rawText = inputField.value.trim();
    
    if (!rawText) {
        showError("Please enter some data.");
        return;
    }

    try {
        // Convert the comma-separated string into an array of numbers
        // e.g. "20, 141, 300" -> [20.0, 141.0, 300.0]
        const features = rawText.split(',').map(item => {
            const num = parseFloat(item.trim());
            if (isNaN(num)) throw new Error(`Invalid number found: "${item}"`);
            return num;
        });

        // 2. Prepare the Payload for SageMaker
        // The format must be: { "instances": [[...]] }
        const payload = {
            instances: [features]
        };

        // 3. Send Request to API Gateway
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        // 4. Handle Response
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`API Error: ${response.status} - ${errorText}`);
        }

        const data = await response.json();
        console.log("Response:", data);

        // 5. Display Result
        // SageMaker Linear Learner usually returns: {"prediction": [{"score": 12345.67}]}
        let predictionValue = "Unknown";
        
        if (data.prediction && data.prediction.predictions) {
             // Structure for some versions of Linear Learner
            predictionValue = data.prediction.predictions[0].score;
        } else if (data.prediction && Array.isArray(data.prediction)) {
             // Alternative structure
            predictionValue = data.prediction[0].score || data.prediction[0];
        } else {
             // Fallback: just dump the raw JSON so you can see it
            predictionValue = JSON.stringify(data);
        }

        // Format as currency if it's a number (assuming housing price)
        if (typeof predictionValue === 'number') {
            predictionOutput.innerText = "$" + predictionValue.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
        } else {
            predictionOutput.innerText = predictionValue;
        }
        
        resultContainer.classList.remove("hidden");

    } catch (err) {
        console.error(err);
        showError(err.message);
    }
}

function showError(msg) {
    const errorContainer = document.getElementById("errorContainer");
    const errorMessage = document.getElementById("errorMessage");
    errorMessage.innerText = msg;
    errorContainer.classList.remove("hidden");
}
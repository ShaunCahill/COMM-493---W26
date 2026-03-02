# Ames Housing Price Predictor

# API Gateway Setup Guide

COMM 493 | Queen's University Smith School of Business

---

# Overview

Amazon API Gateway creates a public URL that your web dashboard can call over the internet. Your Lambda function needs an API Gateway resource and method to be accessible from a browser.

Here is how the pieces fit together:

- Your web dashboard sends a POST request to an API Gateway URL.
- API Gateway forwards the request to the Lambda function.
- The Lambda function calls the SageMaker endpoint and returns the predicted house price.
- API Gateway sends the result back to your web dashboard.

| Lambda Function | API Resource Path | Purpose |
|----------------|-------------------|---------|
| `amesHousing-regression` | `/predict` | House sale price prediction |

---

# Prerequisites

- Lambda function already created and tested (from the Lambda Setup Instructions): `amesHousing-regression`
- AWS account access (AWS Academy Learner Lab)

---

# Part 1: Create the API Gateway

1. Go to the API Gateway console. In the AWS Management Console, type **API Gateway** in the search bar at the top and select it from the results.
2. Click **Create API**.
3. Find the **REST API** card. Make sure it is the standard REST API (not "HTTP API" and not "REST API Private"). Click **Build**.
4. Configure the new API with these settings:
    - Choose: **New API**
    - **API name:** `AmesHousingAPI`
    - **Description:** `API for Ames Housing Price Predictor`
    - **Endpoint Type:** Regional
5. Click **Create API**.

You should now see the API Gateway editor with your new API. The Resources panel on the left will show a single root resource `/`.

---

# Part 2: Create the Predict Resource and Method

## Create the Resource

1. In the API Gateway console, make sure your `AmesHousingAPI` is selected.
2. Click **Create resource**.
3. Configure the new resource:
    - **Resource name:** `predict`
    - The **Resource path** will auto-fill as `/predict`. Leave this as is.
    - Turn on the **CORS (Cross Origin Resource Sharing)** toggle. This is critical for the web dashboard to work. Without CORS enabled, your browser will block requests to the API. Turning on this toggle automatically creates an OPTIONS method on the resource, which handles the browser's CORS preflight requests.
4. Click **Create resource**.

## Create the POST Method

1. With `/predict` selected in the Resources panel, click **Create method**.
2. Configure the method:
    - **Method type:** POST
    - **Integration type:** Lambda
    - Turn on the **Lambda proxy integration** toggle. This passes the full HTTP request (headers, body, query parameters) directly to your Lambda function.
    - **Lambda function:** Select your region (us-east-1) and select `amesHousing-regression` from the dropdown.
3. Click **Create method**.

At this point, your Resources panel should show the root `/` with one child resource: `/predict`. It should have both an **OPTIONS** method (created automatically by the CORS toggle) and a **POST** method.

---

# Part 3: Verify CORS Configuration

When you turned on the **CORS (Cross Origin Resource Sharing)** toggle during resource creation, API Gateway automatically created an OPTIONS method on the resource. This OPTIONS method handles the browser's CORS preflight requests. Let us verify that everything is configured correctly.

1. Select the `/predict` resource in the Resources panel.
2. Verify that there is an **OPTIONS** method listed alongside the POST method.
3. If no OPTIONS method exists, you need to enable CORS manually:
    - Select the resource.
    - Click the **Enable CORS** button in the Resource details section.
    - Under **Access-Control-Allow-Headers**, keep the default value: `Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token`
    - Under **Access-Control-Allow-Origin**, enter: `*`
    - Click **Save**.

> **Note about CORS with Lambda proxy integration:** When you use Lambda proxy integration (which we turned on for the POST method), the Lambda function itself is responsible for including CORS headers in its responses. Your Lambda function already does this. It includes `Access-Control-Allow-Origin: *` in every response. The OPTIONS method created by API Gateway handles the browser's preflight request, and your Lambda function handles the CORS headers on the actual POST responses. Both pieces are needed for CORS to work.

---

# Part 4: Deploy the API

**Important:** Your API will not be accessible until you deploy it. Any time you make changes to your API (adding resources, methods, or CORS settings), you must redeploy for the changes to take effect.

1. Click the **Deploy API** button (in the Resources pane).
2. For **Stage**, select **New stage**.
3. For **Stage name**, type: `prod`
4. Click **Deploy**.
5. After deployment, navigate to **Stages** in the left sidebar and select **prod**.
6. Copy the **Invoke URL** shown in the Stage details section. It will look something like this:

    ```
    https://xxxxxxxxxx.execute-api.us-east-1.amazonaws.com/prod
    ```

Your complete endpoint URL will be:

| Endpoint | URL |
|----------|-----|
| Predict | `https://xxxxxxxxxx.execute-api.us-east-1.amazonaws.com/prod/predict` |

Replace `xxxxxxxxxx` with the actual ID from your Invoke URL.

---

# Part 5: Configure the Web Dashboard

1. Open `index.html` from the `WEB/LINEAR_LEARNER/HTML-JS-CSS/` folder in your web browser. You can double-click the file or drag it into a browser window.
2. Scroll down to the **API Settings** section at the bottom of the page.
3. Paste your endpoint URL into the field:
    - **Prediction API URL:** your Invoke URL + `/predict`
4. Click **Save Settings**. The URL is saved to your browser's local storage, so you will not need to re-enter it each time.
5. Click **Predict Price** to test the connection using the pre-loaded example data.

---

# Part 6: Testing

After saving your API settings, test the dashboard to make sure everything is working.

1. Click one of the **Example** buttons to load sample housing features into the input area.
2. Click **Predict Price**.
3. The progress indicator should show the prediction request being processed.
4. When the prediction is complete, you should see:
    - The predicted sale price displayed as a formatted dollar amount.
    - A result card showing the prediction details.
5. Try pasting your own feature data from the notebook validation set. Copy a row of features (203 comma-separated values) from your notebook and paste it into the input textarea.

---

# Troubleshooting

If something is not working, check the following common issues:

## CORS Error in Browser Console

**Symptom:** The browser console shows a message like "Access to fetch has been blocked by CORS policy."

**Fix:** CORS was not enabled properly. Go back to the API Gateway console and check two things:
1. Verify that the `/predict` resource has an **OPTIONS** method. If not, select the resource, click **Enable CORS**, and save.
2. After making any changes, you must **redeploy** the API. Click **Deploy API**, select the `prod` stage, and click **Deploy**.

## 403 Forbidden

**Symptom:** The API returns a 403 status code.

**Fix:** The API may not be deployed, or you may be using the wrong URL. Go to **Stages** in the left sidebar, select `prod`, and verify the Invoke URL. Make sure your dashboard URL includes the stage name (for example, `.../prod/predict` and not just `.../predict`). If you made any changes after deploying, redeploy the API.

## 502 Bad Gateway

**Symptom:** The API returns a 502 status code.

**Fix:** The Lambda function is encountering an error. Go to the Lambda console, find your function, and check the **Monitor** tab or **CloudWatch Logs** for error details. Common causes include:
- Incorrect `ENDPOINT_NAME` in the Lambda code
- The SageMaker endpoint not being in "InService" status
- The Lambda function not having the correct execution role (must be LabRole)

## 504 Timeout

**Symptom:** The API returns a 504 status code after a long wait.

**Fix:** The Lambda function is taking too long to respond. API Gateway has a maximum timeout of 29 seconds. In the Lambda console, make sure the function timeout is set to 10 seconds or less (go to **Configuration** > **General configuration** > **Edit**). Also verify that the SageMaker endpoint is in "InService" status (not "Creating" or "Failed").

## No Results Showing

**Symptom:** The dashboard does not display any results after clicking Predict Price.

**Fix:** Open the browser developer tools by pressing **F12** and check the **Console** tab for error messages. Verify that the API URL you entered is correct and includes the full path (for example, `.../prod/predict` and not just `.../prod`).

## Mixed Content Error

**Symptom:** The browser console shows a "mixed content" error.

**Fix:** This happens if your dashboard URL starts with `https://` but your API URL starts with `http://`. API Gateway URLs should always use HTTPS. Verify that your Invoke URL begins with `https://`.

---

# Cost Warning

**IMPORTANT:** API Gateway has a generous free tier (1 million API calls per month for the first 12 months). The API Gateway itself will not significantly impact your budget.

However, your **SageMaker endpoint is billed by the hour** (approximately $0.05 to $0.10 per hour). Remember to delete your SageMaker endpoint when you are finished testing. You can delete the endpoint from the SageMaker console under **Inference** > **Endpoints**.

**LEARNER LAB BUDGET WARNING:** If you exceed your Learner Lab budget, your lab account will be disabled and all progress and resources will be permanently lost. SageMaker endpoints are one of the most common causes of unexpected budget consumption. Delete your endpoint as soon as you are done testing.

The Lambda function and API Gateway do not incur meaningful charges when idle. You do not need to delete them to save money.

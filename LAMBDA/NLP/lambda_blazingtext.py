import json
import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# INSTRUCTION: Replace with your BlazingText SageMaker endpoint name
# from the BlazingText Text Classification notebook (STEP 7: Deploy Model)
ENDPOINT_NAME = "your-blazingtext-endpoint-name"

# CORS headers included in every response so the browser allows
# cross-origin requests from the web dashboard.
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST, OPTIONS"
}


def lambda_handler(event, context):
    """
    AWS Lambda function for BlazingText sentiment classification.

    Receives preprocessed review text from the web dashboard,
    forwards it to the BlazingText SageMaker endpoint, and
    returns sentiment predictions (Positive/Negative with confidence).

    Expected input:  { "instances": ["preprocessed text 1", "preprocessed text 2"] }
    Expected output: { "predictions": [{ "label": [...], "prob": [...] }, ...] }
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

    # Prepare and send payload to BlazingText endpoint
    payload = {"instances": instances}
    logger.info("Sending %d instances to endpoint", len(instances))

    try:
        runtime = boto3.client("sagemaker-runtime")
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload)
        )

        result = json.loads(response["Body"].read().decode("utf-8"))
        logger.info("Received %d predictions", len(result))

        return {
            "statusCode": 200,
            "headers": CORS_HEADERS,
            "body": json.dumps({"predictions": result})
        }

    except Exception as e:
        logger.error("SageMaker invocation failed: %s", e)
        return {
            "statusCode": 500,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": f"Endpoint invocation failed: {str(e)}"})
        }

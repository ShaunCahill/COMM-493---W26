import json
import boto3
import csv
import logging
from io import StringIO

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# IMPORTANT: Update this to match your deployed SageMaker endpoint name.
ENDPOINT_NAME = "linear-learner-endpoint"

# Initialize the SageMaker runtime client
runtime = boto3.client("runtime.sagemaker")

# CORS headers required for browser access via API Gateway.
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST, OPTIONS"
}


def convert_to_csv(instances):
    """
    Convert a list of numeric arrays to CSV format for SageMaker Linear Learner.

    SageMaker Linear Learner expects input in text/csv format where each row
    is a comma-separated list of feature values. This function uses the csv
    module for robust conversion (handles edge cases like scientific notation).

    Parameters:
        instances (list): A list of lists, where each inner list contains
                          numeric feature values for one house.

    Returns:
        bytes: The CSV-formatted payload encoded as UTF-8 bytes.
    """
    output = StringIO()
    csv_writer = csv.writer(output)
    csv_writer.writerows(instances)
    return output.getvalue().strip().encode("utf-8")


def lambda_handler(event, context):
    """
    Receives numeric feature arrays from the web dashboard, converts them
    to CSV format, and forwards them to a SageMaker Linear Learner endpoint
    for house price prediction.

    Expected input format (via API Gateway or direct invocation):
    {
        "instances": [[feature1, feature2, ...], [feature1, feature2, ...]]
    }

    Returns predicted sale price(s) from the SageMaker endpoint.
    """
    try:
        # Extract the body from the event (handles both API Gateway and direct invocation).
        body = event.get("body", event)
        if isinstance(body, str):
            body = json.loads(body)

        # Ensure body is a dictionary.
        if not isinstance(body, dict):
            return {
                "statusCode": 400,
                "headers": CORS_HEADERS,
                "body": json.dumps({"error": "Invalid request body format. Expected a JSON object."})
            }

        # Extract instances.
        instances = body.get("instances", [])

        # Validate instances format.
        if not isinstance(instances, list) or len(instances) == 0:
            return {
                "statusCode": 400,
                "headers": CORS_HEADERS,
                "body": json.dumps({"error": "No instances provided. Expected a list of numeric arrays."})
            }

        if not all(isinstance(row, list) for row in instances):
            return {
                "statusCode": 400,
                "headers": CORS_HEADERS,
                "body": json.dumps({"error": "Invalid format for 'instances'. Expected a list of lists."})
            }

        # Convert instances to CSV format.
        csv_payload = convert_to_csv(instances)

        # Log payload details for debugging.
        logger.info(f"Endpoint: {ENDPOINT_NAME}")
        logger.info(f"Number of instances: {len(instances)}")
        logger.info(f"Features per instance: {len(instances[0])}")
        logger.info(f"CSV payload (first 200 chars): {csv_payload[:200]}")

        # Invoke the SageMaker endpoint.
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            Body=csv_payload,
            ContentType="text/csv"
        )

        # Read and parse the response.
        response_body = response["Body"].read().decode("utf-8")
        prediction = json.loads(response_body)

        return {
            "statusCode": 200,
            "headers": CORS_HEADERS,
            "body": json.dumps({"predictions": prediction})
        }

    except Exception as e:
        logger.error(f"Error invoking SageMaker endpoint: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": str(e)})
        }

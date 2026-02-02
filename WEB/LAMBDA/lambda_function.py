import json
import boto3
import os
import logging
import csv
from io import StringIO

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize the SageMaker runtime client
runtime = boto3.client("runtime.sagemaker")

# Load SageMaker endpoint name from environment variable
SAGEMAKER_ENDPOINT = os.environ.get("SAGEMAKER_ENDPOINT", "linear-learner-endpoint")

def convert_to_csv(instances):
    """Convert a list of lists (instances) to CSV format."""
    output = StringIO()
    csv_writer = csv.writer(output)
    csv_writer.writerows(instances)
    return output.getvalue().encode("utf-8")

def lambda_handler(event, context):
    # Define CORS headers to use in every return
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "OPTIONS,POST"
    }

    try:
        logger.info("Received event: %s", json.dumps(event))
        
        # Extract the body from the event
        body = event.get("body")

        # Handle API Gateway Proxy Integration (body is a string)
        if isinstance(body, str):
            body = json.loads(body)
        
        # Ensure body is a dictionary
        if not isinstance(body, dict):
            return {
                "statusCode": 400,
                "headers": headers,
                "body": json.dumps({"error": "Invalid request body format. Expected a JSON object."})
            }
        
        # Extract instances
        instances = body.get("instances", [])

        # Validate instances format
        if not isinstance(instances, list) or not all(isinstance(row, list) for row in instances):
            return {
                "statusCode": 400,
                "headers": headers,
                "body": json.dumps({"error": "Invalid format for 'instances'. Expected a list of lists."})
            }

        # Convert instances to CSV format
        csv_payload = convert_to_csv(instances)

        # Invoke the SageMaker endpoint
        response = runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            Body=csv_payload,
            ContentType="text/csv"
        )

        # Read and parse the response
        response_body = response["Body"].read().decode("utf-8")
        
        # Attempt to parse response as JSON
        try:
            prediction = json.loads(response_body)
        except json.JSONDecodeError:
            prediction = response_body

        return {
            "statusCode": 200,
            "headers": headers,  # <--- THIS IS THE FIX
            "body": json.dumps({"prediction": prediction})
        }

    except boto3.exceptions.botocore.exceptions.ClientError as e:
        logger.error(f"AWS Service Error: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "headers": headers, # <--- THIS IS THE FIX
            "body": json.dumps({"error": "Error communicating with SageMaker Endpoint", "details": str(e)})
        }
        
    except Exception as e:
        logger.error(f"General Error: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "headers": headers, # <--- THIS IS THE FIX
            "body": json.dumps({"error": str(e)})
        }
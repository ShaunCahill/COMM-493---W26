import json
import math
import boto3
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# INSTRUCTION: Replace with your deployed SageMaker KMeans endpoint name
# from the PCA_KMeans HPT notebook (STEP 14: Deploy Best Model).
ENDPOINT_NAME = "your-pca-kmeans-endpoint-name"

# Initialize the SageMaker runtime client
runtime = boto3.client("runtime.sagemaker")

# CORS headers required for browser access via API Gateway.
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST, OPTIONS"
}


# ============================================================
# FEATURE CONFIGURATION
# ============================================================
# The following arrays define the full preprocessing pipeline that
# the notebook applies before sending data to the KMeans endpoint.
# You must extract these values from your notebook and paste them here.
#
# The pipeline is:
#   1. Build a 74-feature vector using quality-conditional medians
#   2. Override with user-provided values (5 features)
#   3. Apply log1p to user-entered skewed features
#   4. Standardize (subtract mean, divide by scale)
#   5. PCA transform (multiply by component matrix) to get 2 values
#   6. Send 2 PCA values as CSV to KMeans endpoint
# ============================================================


# INSTRUCTION: Replace this list with the feature names from your notebook.
# In your notebook after STEP 4 (Preprocess Data), run:
#
#     print(list(df_features.columns))
#
# Then copy the output and paste it here. The order must match exactly.
# The list below is a placeholder showing the expected format.

FEATURE_NAMES = [
    # INSTRUCTION: Paste your feature names here.
    # Example format (your actual list will have ~74 features):
    # "Lot Frontage", "Lot Area", "Overall Qual", "Overall Cond",
    # "Year Built", "Year Remod/Add", "Mas Vnr Area", "BsmtFin SF 1",
    # ...
]


# INSTRUCTION: Replace this dictionary with quality-conditional median values.
# Instead of global medians (which anchor all predictions to the mid-range
# cluster), this dictionary maps each Overall Qual level to its own set of
# median defaults. This way a quality-9 home gets realistic premium defaults
# for correlated features like Exter Qual, Kitchen Qual, Garage Area, etc.
#
# In your notebook after STEP 4 (Preprocess Data), AFTER the log-transform
# has been applied to skewed features, run:
#
#     quality_medians = df_features.groupby('Overall Qual').median()
#     result = {}
#     for qual_level, row in quality_medians.iterrows():
#         result[int(qual_level)] = dict(row)
#     print(json.dumps(result, indent=2))
#
# IMPORTANT: Run this AFTER the log-transform cell in STEP 4. The medians
# must come from the already-log-transformed data. If you extract medians
# before the log-transform, the Lambda will produce incorrect PCA values
# because it would double-log the median defaults.
#
# Then copy the output and paste it here. Each key is a quality level (1-10),
# and each value is a dictionary of feature medians for homes at that quality.

QUALITY_MEDIANS = {
    # INSTRUCTION: Paste your quality-conditional medians here.
    # Example format:
    # 4: {
    #     "Lot Frontage": 3.9512, "Lot Area": 8.9876, "Overall Qual": 4.0,
    #     "Overall Cond": 5.0, "Year Built": 1955, ...
    # },
    # 6: {
    #     "Lot Frontage": 4.2341, "Lot Area": 9.1523, "Overall Qual": 6.0,
    #     "Overall Cond": 5.0, "Year Built": 1976, ...
    # },
    # 9: {
    #     "Lot Frontage": 4.4567, "Lot Area": 9.4321, "Overall Qual": 9.0,
    #     "Overall Cond": 5.0, "Year Built": 2005, ...
    # },
}


# INSTRUCTION: Replace this list with the feature names that were
# log-transformed in your notebook. In your notebook after STEP 4
# (Preprocess Data), the log_transformed_cols variable contains
# this list. Run:
#
#     print(log_transformed_cols)
#
# Then copy the output and paste it here.

LOG_TRANSFORM_COLUMNS = [
    # INSTRUCTION: Paste your log-transformed column names here.
    # Example format:
    # "Lot Frontage", "Lot Area", "Mas Vnr Area", "BsmtFin SF 1",
    # "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF", "1st Flr SF",
    # "2nd Flr SF", "Gr Liv Area", "Garage Area", "Wood Deck SF",
    # "Open Porch SF", "Enclosed Porch", "Screen Porch",
    # ...
]


# INSTRUCTION: Replace this list with the StandardScaler mean values.
# In your notebook after STEP 5 (Standardize Features), run:
#
#     print(list(scaler.mean_))
#
# Then copy the output and paste it here. The order must match FEATURE_NAMES.
# The list should have the same number of elements as FEATURE_NAMES.

SCALER_MEAN = [
    # INSTRUCTION: Paste your scaler mean values here.
    # Example format (one value per feature):
    # 4.1234, 8.9876, 6.0891, 5.5632, 1971.3210, ...
]


# INSTRUCTION: Replace this list with the StandardScaler scale values.
# In your notebook after STEP 5 (Standardize Features), run:
#
#     print(list(scaler.scale_))
#
# Then copy the output and paste it here. The order must match FEATURE_NAMES.

SCALER_SCALE = [
    # INSTRUCTION: Paste your scaler scale values here.
    # Example format (one value per feature):
    # 0.8765, 1.2345, 1.3921, 1.1234, 29.3456, ...
]


# INSTRUCTION: Replace this list with the PCA component matrix.
# In your notebook after STEP 8 (Analyze PCA Loadings), the
# pca_components_matrix variable contains this matrix. Run:
#
#     print([list(row) for row in pca_components_matrix.T])
#
# This prints 2 lists (one per principal component), each with ~74 values.
# Paste the output here. The .T transpose in the print command puts the
# components in the row format the Lambda expects (it loops over components
# as rows and computes the dot product with the feature vector).

PCA_COMPONENTS = [
    # INSTRUCTION: Paste your PCA component vectors here.
    # Format: 2 lists, each with ~74 values.
    # [pc1_loading_feat1, pc1_loading_feat2, ...],  # PC1
    # [pc2_loading_feat1, pc2_loading_feat2, ...],  # PC2
]


# ============================================================
# USER INPUT MAPPING
# ============================================================
# Maps the 5 user-facing input names to their positions in the
# FEATURE_NAMES list. The Lambda receives these 5 values from
# the web dashboard and places them into the full feature vector.
#
# INSTRUCTION: After pasting FEATURE_NAMES above, verify that
# these feature names appear in your list. If your notebook uses
# slightly different column names, update the keys below to match.

USER_FEATURE_NAMES = ["Overall Qual", "Gr Liv Area", "Year Built", "Total Bsmt SF", "Garage Area"]

# USER_FEATURE_MAP is auto-computed from FEATURE_NAMES when the Lambda
# loads. No manual index lookup needed — just populate FEATURE_NAMES above.


# ============================================================
# AUTO-COMPUTE USER_FEATURE_MAP FROM FEATURE_NAMES
# ============================================================
# This runs once at Lambda cold-start. If FEATURE_NAMES is empty
# (student hasn't pasted it yet), the map stays empty and the
# handler returns a clear error message.

USER_FEATURE_MAP = {}
if FEATURE_NAMES:
    for name in USER_FEATURE_NAMES:
        if name in FEATURE_NAMES:
            USER_FEATURE_MAP[name] = FEATURE_NAMES.index(name)
        else:
            logger.warning(f"'{name}' not found in FEATURE_NAMES. Check column names.")


# ============================================================
# PURE PYTHON MATH HELPERS
# ============================================================
# These functions replicate numpy operations without requiring
# a numpy Lambda layer. This keeps the Lambda simple to deploy.
# ============================================================

def log1p(value):
    """
    Compute log(1 + value), equivalent to numpy.log1p().

    Adding 1 before taking the log handles zero values gracefully
    (log(0) is undefined, but log(1) = 0).
    """
    return math.log(1.0 + value)


def standardize(values, means, scales):
    """
    Standardize a feature vector: (x - mean) / scale.

    Equivalent to sklearn StandardScaler.transform().

    Parameters:
        values (list): Raw feature values.
        means (list): Mean for each feature (from scaler.mean_).
        scales (list): Scale for each feature (from scaler.scale_).

    Returns:
        list: Standardized feature values.
    """
    return [(v - m) / s for v, m, s in zip(values, means, scales)]


def pca_transform(standardized, components):
    """
    Project standardized features onto PCA components using matrix multiplication.

    Equivalent to: standardized @ components.T
    where components is a (2 x N) matrix (2 principal components, N features).

    Parameters:
        standardized (list): Standardized feature values (length N).
        components (list): List of 2 lists, each with N values.

    Returns:
        list: 2 PCA values [pc1, pc2].
    """
    result = []
    for component in components:
        dot_product = sum(s * c for s, c in zip(standardized, component))
        result.append(dot_product)
    return result


def get_defaults_for_quality(quality_level):
    """
    Return the median feature vector for the given quality level.

    Uses QUALITY_MEDIANS to find the closest available quality level
    if the exact level is not present in the training data.

    Parameters:
        quality_level (int): Overall quality rating (1-10).

    Returns:
        list: Feature vector with quality-conditional median values.
    """
    if not QUALITY_MEDIANS:
        return [0.0] * len(FEATURE_NAMES)

    # Try exact match first
    qual_int = int(round(quality_level))
    if qual_int in QUALITY_MEDIANS:
        defaults = QUALITY_MEDIANS[qual_int]
    else:
        # Find closest available quality level
        available = sorted(QUALITY_MEDIANS.keys())
        closest = min(available, key=lambda q: abs(q - qual_int))
        defaults = QUALITY_MEDIANS[closest]
        logger.info(f"Quality {qual_int} not in QUALITY_MEDIANS, using closest: {closest}")

    # Build feature vector in FEATURE_NAMES order
    return [defaults.get(name, 0.0) for name in FEATURE_NAMES]


# ============================================================
# LAMBDA HANDLER
# ============================================================

def lambda_handler(event, context):
    """
    Receives 5 property features from the web dashboard, builds a full
    feature vector with quality-conditional median defaults, applies the
    complete preprocessing pipeline (log-transform, standardize, PCA),
    and sends the 2 PCA values to the SageMaker KMeans endpoint for
    cluster assignment.

    Expected input format (via API Gateway or direct invocation):
    {
        "instances": [[overall_qual, gr_liv_area, year_built, total_bsmt_sf, garage_area]]
    }

    Returns cluster assignment and distance to centroid.
    """
    # Handle CORS preflight request
    if event.get("httpMethod") == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": CORS_HEADERS,
            "body": ""
        }

    try:
        # Extract the body from the event (handles both API Gateway and direct invocation)
        body = event.get("body", event)
        if isinstance(body, str):
            body = json.loads(body)

        # Ensure body is a dictionary
        if not isinstance(body, dict):
            return {
                "statusCode": 400,
                "headers": CORS_HEADERS,
                "body": json.dumps({"error": "Invalid request body format. Expected a JSON object."})
            }

        # Extract instances
        instances = body.get("instances", [])

        # Validate instances format
        if not isinstance(instances, list) or len(instances) == 0:
            return {
                "statusCode": 400,
                "headers": CORS_HEADERS,
                "body": json.dumps({"error": "No instances provided. Expected [[qual, area, year, bsmt, garage]]."})
            }

        if not isinstance(instances[0], list) or len(instances[0]) != 5:
            return {
                "statusCode": 400,
                "headers": CORS_HEADERS,
                "body": json.dumps({"error": "Expected 5 feature values: [quality, living_area, year_built, basement_sf, garage_area]."})
            }

        # Validate configuration is populated
        config_checks = [
            (FEATURE_NAMES, "FEATURE_NAMES", "STEP 4 (Preprocess Data): print(list(df_features.columns))"),
            (QUALITY_MEDIANS, "QUALITY_MEDIANS", "STEP 4 (Preprocess Data): see INSTRUCTION comment in this file"),
            (LOG_TRANSFORM_COLUMNS, "LOG_TRANSFORM_COLUMNS", "STEP 4 (Preprocess Data): print(log_transformed_cols)"),
            (SCALER_MEAN, "SCALER_MEAN", "STEP 5 (Standardize): print(list(scaler.mean_))"),
            (SCALER_SCALE, "SCALER_SCALE", "STEP 5 (Standardize): print(list(scaler.scale_))"),
            (PCA_COMPONENTS, "PCA_COMPONENTS", "STEP 8 (PCA Loadings): print([list(row) for row in pca_components_matrix.T])"),
        ]

        for config_val, config_name, notebook_hint in config_checks:
            if not config_val:
                return {
                    "statusCode": 500,
                    "headers": CORS_HEADERS,
                    "body": json.dumps({"error": f"{config_name} is empty. Populate it from your notebook {notebook_hint}"})
                }

        # Validate configuration array lengths are consistent
        num_features = len(FEATURE_NAMES)
        length_checks = [
            (SCALER_MEAN, "SCALER_MEAN"),
            (SCALER_SCALE, "SCALER_SCALE"),
        ]
        for arr, arr_name in length_checks:
            if len(arr) != num_features:
                return {
                    "statusCode": 500,
                    "headers": CORS_HEADERS,
                    "body": json.dumps({"error": f"{arr_name} has {len(arr)} values but FEATURE_NAMES has {num_features}. They must match."})
                }

        if PCA_COMPONENTS and len(PCA_COMPONENTS[0]) != num_features:
            return {
                "statusCode": 500,
                "headers": CORS_HEADERS,
                "body": json.dumps({"error": f"PCA_COMPONENTS[0] has {len(PCA_COMPONENTS[0])} values but FEATURE_NAMES has {num_features}. They must match."})
            }

        if not USER_FEATURE_MAP or len(USER_FEATURE_MAP) != 5:
            return {
                "statusCode": 500,
                "headers": CORS_HEADERS,
                "body": json.dumps({"error": f"USER_FEATURE_MAP could not be computed. Check that these names exist in FEATURE_NAMES: {USER_FEATURE_NAMES}"})
            }

        # Validate input ranges
        user_values = instances[0]
        qual, area, year, bsmt, garage = user_values
        range_checks = [
            (qual, "Overall Quality", 1, 10),
            (area, "Living Area", 1, 100000),
            (year, "Year Built", 1800, 2030),
            (bsmt, "Basement SF", 0, 100000),
            (garage, "Garage Area", 0, 100000),
        ]
        for val, name, lo, hi in range_checks:
            if not (lo <= val <= hi):
                return {
                    "statusCode": 400,
                    "headers": CORS_HEADERS,
                    "body": json.dumps({"error": f"{name} value {val} is out of range [{lo}, {hi}]."})
                }

        quality_level = user_values[0]  # First input is Overall Qual

        logger.info(f"Received 5 user features: {user_values}")

        # ---- Step 1: Build full feature vector with quality-conditional defaults ----
        # Instead of global medians (which anchor all predictions to mid-range),
        # use medians from homes at the same quality level. This ensures that a
        # quality-9 home gets realistic premium defaults for Exter Qual, Kitchen
        # Qual, Year Built, etc.
        feature_vector = get_defaults_for_quality(quality_level)

        logger.info(f"Built {num_features}-feature vector with quality-{int(quality_level)} defaults")

        # ---- Step 2: Override with user-provided values ----
        for i, feat_name in enumerate(USER_FEATURE_NAMES):
            if feat_name in USER_FEATURE_MAP:
                idx = USER_FEATURE_MAP[feat_name]
                feature_vector[idx] = float(user_values[i])

        # ---- Step 3: Apply log1p to user-entered features only ----
        # IMPORTANT: QUALITY_MEDIANS must be extracted from df_features AFTER
        # log-transformation in the notebook (STEP 4). The notebook applies
        # np.log1p() to skewed features before computing medians, so the values
        # in QUALITY_MEDIANS are already log-transformed. Applying log1p again
        # would double-log them, distorting the PCA projection.
        # We only log-transform the 5 raw user inputs (which are in original scale).
        log_col_set = set(LOG_TRANSFORM_COLUMNS)
        for feat_name in USER_FEATURE_NAMES:
            if feat_name in log_col_set and feat_name in USER_FEATURE_MAP:
                idx = USER_FEATURE_MAP[feat_name]
                feature_vector[idx] = log1p(feature_vector[idx])

        # ---- Step 4: Standardize ----
        standardized = standardize(feature_vector, SCALER_MEAN, SCALER_SCALE)

        # ---- Step 5: PCA transform to 2 components ----
        pca_values = pca_transform(standardized, PCA_COMPONENTS)

        logger.info(f"PCA values: PC1={pca_values[0]:.4f}, PC2={pca_values[1]:.4f}")

        # ---- Step 6: Convert to CSV for KMeans endpoint ----
        csv_payload = ",".join([str(v) for v in pca_values])

        logger.info(f"Endpoint: {ENDPOINT_NAME}")
        logger.info(f"CSV payload: {csv_payload}")

        # ---- Step 7: Invoke the SageMaker KMeans endpoint ----
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            Body=csv_payload.encode("utf-8"),
            ContentType="text/csv"
        )

        # ---- Step 8: Parse and return the response ----
        response_body = response["Body"].read().decode("utf-8")
        prediction = json.loads(response_body)

        logger.info(f"KMeans response: {response_body}")

        return {
            "statusCode": 200,
            "headers": CORS_HEADERS,
            "body": json.dumps({"predictions": prediction})
        }

    except Exception as e:
        logger.error(f"Error in PCA-KMeans pipeline: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": str(e)})
        }

#!/usr/bin/env python3
"""
Generate PCA + KMeans HPT notebooks for Demo 1 and Demo 2.

Reads base notebooks from PCA_KMEANS/DEMO 1/ and PCA_KMEANS/DEMO 2/,
creates HPT versions that:
- Keep Steps 1-8 identical (import, setup, load, preprocess, standardize, train PCA, scree, transform)
- Step 9: Determine optimal K via sklearn silhouette analysis
- Step 10: Replace manual KMeans training with SageMaker HPT using the dynamic K
- Adapt visualization, deploy, predict, cleanup steps
"""

import json
import copy
import os
import shutil

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(REPO_ROOT)  # Go up one more to GITHUB/

def make_cell(cell_type, source, cell_id=None):
    """Create a notebook cell."""
    import uuid
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source if isinstance(source, list) else source.split('\n') if '\n' not in source else [source],
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    cell["id"] = cell_id or str(uuid.uuid4())[:12]
    return cell


def make_md(text, cell_id=None):
    """Create a markdown cell from a multiline string."""
    lines = text.strip().split('\n')
    source = [line + '\n' for line in lines[:-1]] + [lines[-1]]
    return make_cell("markdown", source, cell_id)


def make_code(text, cell_id=None):
    """Create a code cell from a multiline string."""
    lines = text.strip().split('\n')
    source = [line + '\n' for line in lines[:-1]] + [lines[-1]]
    return make_cell("code", source, cell_id)


# ============================================================================
# HPT-specific cells (shared structure, parameterized by demo number)
# ============================================================================

def get_determine_k_md(demo_num):
    """Step 9 markdown: DETERMINE OPTIMAL K"""
    return make_md(f"""# STEP 9: DETERMINE OPTIMAL K

Before tuning the KMeans training hyperparameters, we first need to determine **how many clusters (K)** to use. SageMaker KMeans does not support K as a tunable hyperparameter in its tuning jobs, so we determine K locally using a quick sklearn-based Silhouette Score analysis on the PCA-reduced data.

This is the same approach used in the base Demo {demo_num} notebook (Elbow Plot + Silhouette Score), but condensed into a single function. The best K found here will be passed as a fixed parameter to the HPT step that follows.

**What is about to happen:**
- Run sklearn KMeans for K=2 through K=7 on the PCA-reduced data
- Compute the Silhouette Score for each K
- Select the K with the highest Silhouette Score
- Plot the Silhouette Score vs K to visualize the choice""")


def get_determine_k_code():
    """Step 9 code: determine optimal K via silhouette analysis"""
    return make_code("""def determine_optimal_k_local(pca_np, k_range=range(2, 8)):
    \"\"\"
    Use sklearn KMeans + Silhouette Score to find optimal K locally.

    Runs KMeans for each K in k_range, computes the Silhouette Score,
    and returns the K with the highest score.

    Parameters:
        pca_np (np.ndarray): PCA-transformed data as float32 array.
        k_range (range): Range of K values to evaluate.

    Returns:
        int: The best K value.
    \"\"\"
    from sklearn.cluster import KMeans as SklearnKMeans
    from sklearn.metrics import silhouette_score

    scores = []
    for k in k_range:
        km = SklearnKMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(pca_np)
        score = silhouette_score(pca_np, labels)
        scores.append(score)
        print(f"K={k}: Silhouette Score = {score:.4f}")

    best_idx = np.argmax(scores)
    best_k = list(k_range)[best_idx]
    print(f"\\nBest K: {best_k} (Silhouette = {scores[best_idx]:.4f})")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(list(k_range), scores, marker='o', linewidth=2, color='steelblue')
    ax.set_xlabel('Number of Clusters (K)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score vs K')
    ax.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'Best K={best_k}')
    ax.legend()
    plt.tight_layout()
    plt.show()

    return best_k


pca_np = pca_df.values.astype('float32')
best_k = determine_optimal_k_local(pca_np)""")


def get_determine_k_wtlf():
    """Step 9 What to Look For"""
    return make_md("""## What to Look For

**Silhouette Score plot:** The best K is the one with the highest Silhouette Score. A score closer to 1.0 means clusters are well-separated and cohesive. A score near 0 means clusters overlap significantly.

**Typical result:** For the Ames Housing data, K=3 often scores highest (corresponding to budget, mid-range, and premium housing tiers), but your result may differ depending on the encoding strategy.

**This K will be used in the next step** as a fixed parameter when the HPT job tunes the other KMeans training hyperparameters (mini_batch_size, extra_center_factor, epochs, init_method).""")


def get_hpt_training_md(demo_num, num_features_approx):
    """Step 10 markdown: TRAIN KMEANS WITH HYPERPARAMETER TUNING"""
    return make_md(f"""# STEP 10: TRAIN KMEANS MODEL WITH HYPERPARAMETER TUNING

In the base Demo {demo_num} notebook, we manually trained KMeans models with K=2 through K=7. In this notebook, we determined the optimal K in STEP 9 using Silhouette Score analysis. Now we use **SageMaker Automatic Model Tuning (Hyperparameter Tuning)** to optimize the remaining KMeans training hyperparameters for that K.

SageMaker runs multiple training jobs with different hyperparameter combinations and identifies the best configuration using a Bayesian optimization strategy.

**What is about to happen:**
- Split the PCA-reduced data into **train (80%)** and **test (20%)** sets
- Configure a KMeans estimator with K fixed to the value determined in STEP 9
- Define hyperparameter ranges for SageMaker to search
- Launch a tuning job with 10 total training jobs (2 running in parallel at a time)

**Fixed hyperparameter:**
- `k` = best_k (determined dynamically in STEP 9 via Silhouette Score analysis)

**Tuned hyperparameters:**

| Hyperparameter | Type | Range | Why Tune It? |
|----------------|------|-------|-------------|
| `mini_batch_size` | Integer | 100 - 2,000 | Controls how many data points are used in each training iteration. Smaller batches can find better local optima but take longer. |
| `extra_center_factor` | Integer | 2 - 10 | SageMaker initially creates K x extra_center_factor centers, then reduces to K. Higher values explore more of the data space during initialization. |
| `epochs` | Integer | 1 - 10 | Number of passes through the training data. More epochs allow the algorithm to refine centroids further. |
| `init_method` | Categorical | random, kmeans++ | How initial cluster centers are chosen. kmeans++ spreads centers apart, random picks randomly. |

**Objective metric:** `test:msd` (Mean Squared Distance) - measures the average squared distance from each test point to its nearest cluster center. **Lower is better** (tighter clusters).

**AWS cost note:** This step launches 10 training jobs (2 at a time). Each uses `ml.m5.large` and completes in under a minute, keeping costs minimal.""")


def get_hpt_training_code(demo_num, prefix_name):
    """Step 10 code: HPT training"""
    return make_code(f"""def train_kmeans_with_hpt(pca_np, num_pca_components, k, sagemaker_session, role, bucket, prefix, sagemaker_client):
    \"\"\"
    Train KMeans using SageMaker Hyperparameter Tuning.

    Splits the PCA-reduced data into train/test, creates a KMeans estimator
    with the given K, defines hyperparameter ranges, and launches a tuning job.

    Parameters:
        pca_np (np.ndarray): PCA-transformed data as float32 array.
        num_pca_components (int): Number of PCA components (feature_dim for KMeans).
        k (int): Number of clusters (determined in STEP 9).
        sagemaker_session: The SageMaker session.
        role (str): The IAM role ARN.
        bucket (str): The S3 bucket name.
        prefix (str): The S3 prefix.
        sagemaker_client: The Boto3 SageMaker client.

    Returns:
        HyperparameterTuner: The completed tuner object.
    \"\"\"
    from sagemaker.tuner import HyperparameterTuner, IntegerParameter, CategoricalParameter
    from sklearn.model_selection import train_test_split

    # Split PCA data into train (80%) and test (20%) for the tuning metric
    train_data, test_data = train_test_split(pca_np, test_size=0.2, random_state=42)
    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')

    print(f"Train set: {{train_data.shape[0]}} samples")
    print(f"Test set:  {{test_data.shape[0]}} samples")
    print(f"K:         {{k}} clusters (from STEP 9)")
    print(f"Features:  {{num_pca_components}} PCA components\\n")

    # Create the KMeans estimator with fixed hyperparameters
    output_path = f's3://{{bucket}}/{{prefix}}/kmeans-hpt/output'

    kmeans = sagemaker.KMeans(
        sagemaker_session=sagemaker_session,
        role=role,
        instance_count=1,
        instance_type='ml.m5.large',
        k=k,
        output_path=output_path,
        feature_dim=num_pca_components
    )

    # Define hyperparameter ranges to tune
    hyperparameter_ranges = {{
        'mini_batch_size': IntegerParameter(100, 2000),
        'extra_center_factor': IntegerParameter(2, 10),
        'epochs': IntegerParameter(1, 10),
        'init_method': CategoricalParameter(['random', 'kmeans++']),
    }}

    # Create the Hyperparameter Tuner
    tuner = HyperparameterTuner(
        estimator=kmeans,
        objective_metric_name='test:msd',
        hyperparameter_ranges=hyperparameter_ranges,
        objective_type='Minimize',
        max_jobs=10,
        max_parallel_jobs=2,
    )

    # Create RecordSet objects for train and test channels
    train_records = kmeans.record_set(train_data, channel='train')
    test_records = kmeans.record_set(test_data, channel='test')

    # Launch the tuning job
    print("Launching hyperparameter tuning job...")
    print("This will run 10 training jobs (2 at a time). Estimated time: 10-15 minutes.\\n")
    tuner.fit([train_records, test_records])
    print("\\nHyperparameter tuning job complete!")

    return tuner


tuner = train_kmeans_with_hpt(
    pca_np, num_pca_components, best_k,
    sagemaker_session, role, bucket, prefix,
    sagemaker_client
)""")


def get_hpt_training_wtlf():
    """Step 10 What to Look For"""
    return make_md("""## What to Look For

You should see the tuning job launch and complete with 10 training jobs. SageMaker uses **Bayesian optimization** to intelligently choose hyperparameter combinations, learning from earlier jobs to focus on promising regions of the search space.

**While it runs:** You can monitor progress in the AWS Console under SageMaker > Training > Hyperparameter tuning jobs.

**When it finishes:** The tuner automatically identifies the best training job (lowest test:msd) for the K value determined in STEP 9. We will analyze the results in the next step.""")


def get_analyze_results_md():
    """Step 11 markdown: ANALYZE TUNING RESULTS"""
    return make_md("""# STEP 11: ANALYZE TUNING RESULTS

Now we examine what the tuner found. This step retrieves the best hyperparameters and displays the full tuning analytics so you can see how different configurations performed.

**What is about to happen:**
- Retrieve the best training job name and its hyperparameters
- Display a comparison table of all 10 training jobs sorted by test:msd
- Show the best hyperparameter values that minimized mean squared distance""")


def get_analyze_results_code(demo_num, prefix_name):
    """Step 11 code: analyze tuning results"""
    return make_code(f"""def analyze_tuning_results(tuner, sagemaker_client):
    \"\"\"
    Retrieve and display the results of the hyperparameter tuning job.

    Shows the best hyperparameters and a comparison table of all training jobs.

    Parameters:
        tuner: The completed HyperparameterTuner.
        sagemaker_client: The Boto3 SageMaker client.

    Returns:
        dict: The best training job description including model artifact S3 URI.
    \"\"\"
    # Get the tuning job name
    tuning_job_name = tuner.latest_tuning_job.name
    print(f"Tuning Job: {{tuning_job_name}}\\n")

    # Get tuning job results
    tuning_results = sagemaker_client.describe_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuning_job_name
    )

    # Best training job
    best_job = tuning_results['BestTrainingJob']
    best_job_name = best_job['TrainingJobName']
    best_metric = best_job['FinalHyperParameterTuningJobObjectiveMetric']['Value']

    print(f"Best Training Job: {{best_job_name}}")
    print(f"Best test:msd: {{best_metric:.6f}}\\n")

    # Display best hyperparameters
    best_hp = best_job['TunedHyperParameters']
    print("Best Hyperparameters:")
    print(f"  mini_batch_size:    {{best_hp.get('mini_batch_size', 'N/A')}}")
    print(f"  extra_center_factor: {{best_hp.get('extra_center_factor', 'N/A')}}")
    print(f"  epochs:             {{best_hp.get('epochs', 'N/A')}}")
    print(f"  init_method:        {{best_hp.get('init_method', 'N/A')}}\\n")

    # Get all training jobs for comparison
    all_jobs = sagemaker_client.list_training_jobs_for_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=tuning_job_name,
        SortBy='FinalObjectiveMetricValue',
        SortOrder='Ascending'
    )['TrainingJobSummaries']

    # Build comparison table
    rows = []
    for i, job in enumerate(all_jobs):
        job_hp = job.get('TunedHyperParameters', {{}})
        metric_val = job.get('FinalHyperParameterTuningJobObjectiveMetric', {{}}).get('Value', 'N/A')
        status = job.get('TrainingJobStatus', 'Unknown')
        rows.append({{
            'Rank': i + 1,
            'mini_batch_size': job_hp.get('mini_batch_size', 'N/A'),
            'extra_center_factor': job_hp.get('extra_center_factor', 'N/A'),
            'epochs': job_hp.get('epochs', 'N/A'),
            'init_method': job_hp.get('init_method', 'N/A'),
            'test:msd': f"{{metric_val:.6f}}" if isinstance(metric_val, float) else metric_val,
            'Status': status
        }})

    results_df = pd.DataFrame(rows)
    print("All Training Jobs (sorted by test:msd, lower is better):")
    print(tabulate(results_df.values.tolist(), headers=results_df.columns.tolist(), tablefmt='grid'))

    # Get best model artifact URI
    best_job_desc = sagemaker_client.describe_training_job(TrainingJobName=best_job_name)
    best_model_uri = best_job_desc['ModelArtifacts']['S3ModelArtifacts']
    print(f"\\nBest model artifact: {{best_model_uri}}")

    return best_job_desc


best_job_desc = analyze_tuning_results(tuner, sagemaker_client)
best_model_uri = best_job_desc['ModelArtifacts']['S3ModelArtifacts']

# Store as a list for compatibility with the visualization function
kmeans_model_paths = [(best_k, best_model_uri)]""")


def get_analyze_results_wtlf():
    """Step 11 What to Look For"""
    return make_md("""## What to Look For

**Best hyperparameters:** Look at which combination of mini_batch_size, extra_center_factor, epochs, and init_method produced the lowest test:msd (mean squared distance). Lower MSD means data points are closer to their assigned cluster centers on average.

**Comparison table:** Notice how different hyperparameter combinations affect the MSD. Some patterns to look for:
- Does `kmeans++` initialization consistently outperform `random`?
- Do more epochs always help, or is there diminishing returns?
- How sensitive is the result to `mini_batch_size`?

**Bayesian optimization:** The tuner does not search randomly. It uses results from earlier jobs to focus on promising regions. You may notice that later jobs (higher rank numbers) tend to have worse scores, because the tuner already found good values early.

**Compare with base notebook:** In the base Demo notebook, you manually searched K values. Here, K was determined automatically in STEP 9, and the tuner optimized the training configuration for that K. Both approaches are valid and complementary.""")


def get_visualize_clusters_md_hpt(demo_num, num_features_approx):
    """Step 12 markdown: VISUALIZE CLUSTERS (HPT version)"""
    return make_md(f"""# STEP 12: VISUALIZE CLUSTERS

This is the payoff of the entire notebook. We visualize the cluster assignments from the best tuned model and check whether they correspond to meaningful housing price tiers.

**What is about to happen:**
- Download the best model's centroids from S3
- Assign each data point to its nearest centroid
- **Plot 1 (left):** Scatter plot of all homes in PCA space, colored by cluster assignment with centroids marked as black X
- **Plot 2 (right):** Same scatter plot, but colored by SalePrice (green = expensive, red = cheap). This is the validation: we never told the model about SalePrice, so if the color gradient aligns with the clusters, it means the algorithm found real structure.
- **Plot 3:** Box plot showing SalePrice distribution within each cluster
- **Cluster profile table:** Average SalePrice, quality, size, and age per cluster with plain-English descriptions

**Note:** The visualization uses the **full dataset** (not just the training split from STEP 10). The train/test split was only needed for the tuning metric. For visualization and profiling, we want to see all data points.""")


def get_deploy_md_hpt(demo_num):
    """Deploy markdown for HPT version"""
    step_num = 13
    return make_md(f"""# STEP {step_num}: DEPLOY BEST KMEANS MODEL TO ENDPOINT

Deploy the best KMeans model from the tuning job to a real-time SageMaker endpoint for predictions. This allows us to send new PCA-transformed data points and receive cluster assignments.

**What is about to happen:**
- Delete any existing endpoint with the same name (to avoid conflicts)
- Create a SageMaker Model object from the best tuned model artifacts
- Deploy it to an `ml.m5.large` instance as a real-time endpoint
- This may take 3-5 minutes while AWS provisions the instance""")


def get_deploy_code_hpt(demo_num, endpoint_name_val):
    """Deploy code for HPT version"""
    return make_code(f"""def delete_endpoint_and_config(endpoint_name, sagemaker_client):
    \"\"\"
    Delete a SageMaker endpoint and its endpoint configuration if they exist.
    Polls until both resources are fully deleted.

    This function is defined here in the deploy step and reused in the
    cleanup step (STEP 15) without duplication.

    Parameters:
        endpoint_name (str): The name of the endpoint to delete.
        sagemaker_client: The Boto3 SageMaker client.
    \"\"\"
    # Delete endpoint if it exists
    try:
        sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        print(f'Deleting endpoint: {{endpoint_name}}')
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
    except ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException' and 'Could not find' in e.response['Error']['Message']:
            print(f'Endpoint "{{endpoint_name}}" does not exist.')
        else:
            raise

    # Delete endpoint config if it exists
    try:
        sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_name)
        print(f'Deleting endpoint configuration: {{endpoint_name}}')
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
    except ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException' and 'Could not find' in e.response['Error']['Message']:
            print(f'Endpoint config "{{endpoint_name}}" does not exist.')
        else:
            raise

    # Poll for deletion
    print('Waiting for resources to be deleted...')
    for _ in range(30):
        endpoint_exists = True
        config_exists = True

        try:
            sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        except ClientError:
            endpoint_exists = False

        try:
            sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_name)
        except ClientError:
            config_exists = False

        if not endpoint_exists and not config_exists:
            print('Endpoint and endpoint config fully deleted.')
            break

        time.sleep(10)
    else:
        print('Warning: resources may not be fully deleted after 30 checks.')


def deploy_best_tuned_model(best_model_uri, best_k, endpoint_name, kmeans_image_uri, role, sagemaker_session, sagemaker_client):
    \"\"\"
    Deploy the best model from the tuning job to a SageMaker endpoint.

    Parameters:
        best_model_uri (str): S3 URI of the best model artifacts.
        best_k (int): The number of clusters.
        endpoint_name (str): The name for the endpoint.
        kmeans_image_uri (str): The KMeans Docker image URI.
        role (str): The IAM role ARN.
        sagemaker_session: The SageMaker session.
        sagemaker_client: The Boto3 SageMaker client.

    Returns:
        Predictor: The SageMaker Predictor object for making predictions.
    \"\"\"
    # Delete any existing endpoint first
    delete_endpoint_and_config(endpoint_name, sagemaker_client)

    # Create a SageMaker Model object from the best tuned model
    best_model = Model(
        model_data=best_model_uri,
        image_uri=kmeans_image_uri,
        role=role,
        sagemaker_session=sagemaker_session
    )

    # Deploy the model to an endpoint
    print(f'\\nDeploying best tuned KMeans model (K={{best_k}}) to endpoint: {{endpoint_name}}')
    print('This may take several minutes...')

    predictor = best_model.deploy(
        endpoint_name=endpoint_name,
        initial_instance_count=1,
        instance_type='ml.m5.large'
    )

    print(f'\\nEndpoint "{{endpoint_name}}" deployed successfully!')
    return predictor


endpoint_name = '{endpoint_name_val}'
predictor = deploy_best_tuned_model(
    best_model_uri, best_k, endpoint_name,
    kmeans_image_uri, role, sagemaker_session, sagemaker_client
)

# Configure the predictor serializer for CSV input
predictor = Predictor(endpoint_name=endpoint_name, serializer=CSVSerializer())""")


def get_deploy_wtlf_hpt(endpoint_name_val):
    """Deploy What to Look For for HPT version"""
    return make_md(f"""## What to Look For

You should see the endpoint being created and eventually a success message. The endpoint name is `{endpoint_name_val}`.

**Important:** The endpoint will continue incurring charges until you delete it in STEP 15. Do not skip the cleanup step.""")


def get_predict_md_hpt(demo_num):
    """Interactive prediction markdown for HPT version"""
    extra_log = ""
    if demo_num == 2:
        extra_log = "\n- The code applies: log-transformation (for skewed features) then standardization (StandardScaler) then PCA transformation then KMeans prediction"
    else:
        extra_log = "\n- The code applies: standardization (StandardScaler) then PCA transformation then KMeans prediction"

    log_note = ""
    if demo_num == 2:
        log_note = """

**Why the extra log-transform step matters:** Because we log-transformed certain features during training (Step 4), we must apply the same transformation to new predictions. The StandardScaler was fitted on log-transformed data, so feeding it raw values would produce incorrect scaled values, which would produce incorrect PCA coordinates, which would produce incorrect cluster assignments. This is the cost of a more sophisticated pipeline, but the better cluster quality is worth it. Compare this to Demo 1, where prediction only requires standardization and PCA (no log step)."""

    return make_md(f"""# STEP 14: INTERACTIVE PREDICTION

This step demonstrates the full real-world ML pipeline: a user enters raw housing feature values (Overall Quality, Living Area, etc.), and the code applies the same preprocessing, standardization, and PCA transformation before sending the result to the KMeans endpoint.

**What is about to happen:**
- First, three pre-built example homes run automatically to show how different homes land in different clusters
- Then, you will be prompted to enter 5 key housing features (quality, size, year, basement, garage)
- All other features are filled with median values from the training data{extra_log}
- The result shows which cluster the home belongs to, along with the cluster profile{log_note}

**The automatic examples include:**
- A luxury home (Quality=9, 3,000 sq ft, built 2010)
- A starter home (Quality=5, 900 sq ft, built 1975)
- A fixer-upper (Quality=3, 1,100 sq ft, built 1950)

**Then try your own scenarios:**
- Your dream home: enter whatever values you like!

Type `quit` at any prompt to exit.""")


def get_cleanup_md_hpt():
    """Cleanup markdown for HPT version"""
    return make_md("""# STEP 15: CLEANUP

Delete the SageMaker endpoint and endpoint configuration to stop incurring charges. This reuses the `delete_endpoint_and_config` function defined in STEP 13.

**Important:** SageMaker endpoints charge by the hour. Always run this cell when you are done to avoid unexpected costs on your AWS account.""")


def get_cleanup_code():
    """Cleanup code cell"""
    return make_code("""# ⚠️ AWS COST WARNING ⚠️
# SageMaker endpoints incur charges as long as they are running.
# Make sure to delete your endpoint when you are done to avoid unexpected costs.

response = input("Are you sure you want to delete the endpoint? (yes/no): ").strip().lower()
if response == 'yes':
    delete_endpoint_and_config(endpoint_name, sagemaker_client)
    print("\\nEndpoint cleanup complete.")
else:
    print("\\nEndpoint was NOT deleted. Remember to delete it later to avoid charges.")""")


# ============================================================================
# Generation functions
# ============================================================================

def generate_demo1_hpt():
    """Generate Demo 1 HPT notebook."""
    base_path = os.path.join(REPO_ROOT, "PCA_KMEANS", "DEMO 1", "PCA_KMeans_Demo1_Standard_Encoding.ipynb")
    with open(base_path) as f:
        base_nb = json.load(f)

    base_cells = base_nb['cells']
    cells = []

    # Cell 0: Modified intro
    intro = make_md("""# PCA + KMeans Unsupervised Learning Demo

## Demo 1: Standard Encoding (Frequency + One-Hot) - Hyperparameter Tuning

This notebook extends the base Demo 1 notebook by adding **SageMaker Hyperparameter Tuning** to the KMeans clustering step. Instead of manually training KMeans models with different K values and comparing Elbow Plots and Silhouette Scores, we use SageMaker's Automatic Model Tuning to systematically optimize KMeans training hyperparameters.

**What is different from the base Demo 1 notebook:**
- Steps 1-8 are identical (data loading, preprocessing, PCA)
- Step 9 determines optimal K via Silhouette Score analysis
- Step 10 uses SageMaker Hyperparameter Tuning to optimize training for that K
- Step 11 analyzes the tuning results
- Steps 12-15 are adapted to use the best tuned model

**Prerequisites:** Run the base Demo 1 notebook first to understand the manual approach. This notebook builds on that foundation.

**Encoding strategy:** Frequency encoding for rare categories + one-hot encoding for categorical features, producing ~200 sparse features. This is the same encoding as the base Demo 1 notebook.

**Compare with Demo 2 HPT:** Demo 2 uses optimized encoding (~74 dense features) with the same HPT approach. Running both shows how encoding choices affect tuning results.""")
    cells.append(intro)

    # Cells 1-25: Copy from base (Steps 1-8, through PCA transform)
    # Cell 1 = Step 1 markdown
    # Cell 2 = Step 1 code (imports/install)
    # ...
    # Cell 25 = Step 8 What to Look For
    for i in range(1, 26):
        cell = copy.deepcopy(base_cells[i])
        cells.append(cell)

    # Step 9: DETERMINE OPTIMAL K
    cells.append(get_determine_k_md(1))
    cells.append(get_determine_k_code())
    cells.append(get_determine_k_wtlf())

    # Step 10: TRAIN KMEANS WITH HYPERPARAMETER TUNING
    cells.append(get_hpt_training_md(1, 200))
    cells.append(get_hpt_training_code(1, "pca-kmeans-demo1"))
    cells.append(get_hpt_training_wtlf())

    # Step 11: ANALYZE TUNING RESULTS
    cells.append(get_analyze_results_md())
    cells.append(get_analyze_results_code(1, "pca-kmeans-demo1"))
    cells.append(get_analyze_results_wtlf())

    # Step 12: VISUALIZE CLUSTERS (copy from base cell 32-35, adapted)
    cells.append(get_visualize_clusters_md_hpt(1, 200))
    # Copy the visualization code cell from base (cell 33)
    viz_cell = copy.deepcopy(base_cells[33])
    cells.append(viz_cell)
    # Copy the What to Look For (cell 34)
    cells.append(copy.deepcopy(base_cells[34]))
    # Copy the Business Application (cell 35)
    cells.append(copy.deepcopy(base_cells[35]))

    # Step 13: DEPLOY
    cells.append(get_deploy_md_hpt(1))
    cells.append(get_deploy_code_hpt(1, "ames-pca-kmeans-demo1-hpt"))
    cells.append(get_deploy_wtlf_hpt("ames-pca-kmeans-demo1-hpt"))

    # Step 14: INTERACTIVE PREDICTION
    cells.append(get_predict_md_hpt(1))
    # Copy prediction code from base (cell 40)
    cells.append(copy.deepcopy(base_cells[40]))

    # Step 15: CLEANUP
    cells.append(get_cleanup_md_hpt())
    cells.append(get_cleanup_code())

    # Build notebook
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": copy.deepcopy(base_nb.get("metadata", {})),
        "cells": cells
    }

    out_path = os.path.join(REPO_ROOT, "HYPER_PARAMETER_TUNING", "PCA_KMEANS", "DEMO 1",
                            "PCA_KMeans_Demo1_Standard_Encoding_HPT.ipynb")
    with open(out_path, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"Generated: {out_path}")
    print(f"  Cells: {len(cells)} ({sum(1 for c in cells if c['cell_type'] == 'code')} code, "
          f"{sum(1 for c in cells if c['cell_type'] == 'markdown')} markdown)")
    return out_path


def generate_demo2_hpt():
    """Generate Demo 2 HPT notebook."""
    base_path = os.path.join(REPO_ROOT, "PCA_KMEANS", "DEMO 2", "PCA_KMeans_Demo2_Optimized_Encoding.ipynb")
    with open(base_path) as f:
        base_nb = json.load(f)

    base_cells = base_nb['cells']
    cells = []

    # Cell 0: Modified intro
    intro = make_md("""# PCA + KMeans Unsupervised Learning Demo

## Demo 2: Optimized Encoding (Ordinal + Target) - Hyperparameter Tuning

This notebook extends the base Demo 2 notebook by adding **SageMaker Hyperparameter Tuning** to the KMeans clustering step. Instead of manually training KMeans models and then running a local grid search across PCA dimensions and K values, we use SageMaker's Automatic Model Tuning to systematically optimize KMeans training hyperparameters.

**What is different from the base Demo 2 notebook:**
- Steps 1-8 are identical (data loading, optimized preprocessing with ordinal + target encoding, PCA)
- Step 9 determines optimal K via Silhouette Score analysis
- Step 10 uses SageMaker Hyperparameter Tuning to optimize training for that K (replacing both the manual K-search and grid search from the base notebook)
- Step 11 analyzes the tuning results
- Steps 12-15 are adapted to use the best tuned model

**Prerequisites:** Run the base Demo 2 notebook first to understand the manual approach and the grid search. This notebook builds on that foundation.

**Encoding strategy:** Ordinal encoding for ranked features (20 columns) + target encoding for nominal features (18 columns with mean SalePrice), producing ~74 dense features. This gives PCA much richer inputs than Demo 1's ~200 sparse one-hot features.

**Compare with Demo 1 HPT:** Demo 1 uses standard encoding (~200 sparse features) with the same HPT approach. Running both shows how encoding choices affect tuning results.""")
    cells.append(intro)

    # Cells 1-25: Copy from base (Steps 1-8, through PCA transform)
    for i in range(1, 26):
        cell = copy.deepcopy(base_cells[i])
        cells.append(cell)

    # Step 9: DETERMINE OPTIMAL K
    cells.append(get_determine_k_md(2))
    cells.append(get_determine_k_code())
    cells.append(get_determine_k_wtlf())

    # Step 10: TRAIN KMEANS WITH HYPERPARAMETER TUNING
    cells.append(get_hpt_training_md(2, 74))
    cells.append(get_hpt_training_code(2, "pca-kmeans-demo2"))
    cells.append(get_hpt_training_wtlf())

    # Step 11: ANALYZE TUNING RESULTS
    cells.append(get_analyze_results_md())
    cells.append(get_analyze_results_code(2, "pca-kmeans-demo2"))
    cells.append(get_analyze_results_wtlf())

    # Step 12: VISUALIZE CLUSTERS (copy from base cell 32-33 + adapted what-to-look-for)
    cells.append(get_visualize_clusters_md_hpt(2, 74))
    # Copy the visualization code cell from base (cell 33)
    viz_cell = copy.deepcopy(base_cells[33])
    cells.append(viz_cell)
    # Copy the What to Look For from Demo 2 base (cell 41 - post grid search)
    cells.append(copy.deepcopy(base_cells[41]))
    # Copy the Business Application (cell 42)
    cells.append(copy.deepcopy(base_cells[42]))

    # Step 13: DEPLOY
    cells.append(get_deploy_md_hpt(2))
    cells.append(get_deploy_code_hpt(2, "ames-pca-kmeans-demo2-hpt"))
    cells.append(get_deploy_wtlf_hpt("ames-pca-kmeans-demo2-hpt"))

    # Step 14: INTERACTIVE PREDICTION
    cells.append(get_predict_md_hpt(2))
    # Copy prediction code from base (cell 47)
    cells.append(copy.deepcopy(base_cells[47]))

    # Step 15: CLEANUP
    cells.append(get_cleanup_md_hpt())
    cells.append(get_cleanup_code())

    # Build notebook
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": copy.deepcopy(base_nb.get("metadata", {})),
        "cells": cells
    }

    out_path = os.path.join(REPO_ROOT, "HYPER_PARAMETER_TUNING", "PCA_KMEANS", "DEMO 2",
                            "PCA_KMeans_Demo2_Optimized_Encoding_HPT.ipynb")
    with open(out_path, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"Generated: {out_path}")
    print(f"  Cells: {len(cells)} ({sum(1 for c in cells if c['cell_type'] == 'code')} code, "
          f"{sum(1 for c in cells if c['cell_type'] == 'markdown')} markdown)")
    return out_path


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Generating PCA + KMeans HPT notebooks...\n")

    d1_path = generate_demo1_hpt()
    print()
    d2_path = generate_demo2_hpt()

    # Copy AmesHousing.csv to HPT directories
    for demo in ["DEMO 1", "DEMO 2"]:
        src = os.path.join(REPO_ROOT, "PCA_KMEANS", demo, "AmesHousing.csv")
        dst = os.path.join(REPO_ROOT, "HYPER_PARAMETER_TUNING", "PCA_KMEANS", demo, "AmesHousing.csv")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"Copied AmesHousing.csv to HYPER_PARAMETER_TUNING/PCA_KMEANS/{demo}/")

    print("\nDone! Generated notebooks:")
    print(f"  1. {d1_path}")
    print(f"  2. {d2_path}")

from flask import Flask, request, jsonify, send_from_directory, session
from azure.storage.blob import BlobServiceClient
import pandas as pd
import io
import requests
from ydata_profiling import ProfileReport
import os
from dotenv import load_dotenv
import time
import secrets
from flask_session import Session
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

app.config["SESSION_TYPE"] = "filesystem"
app.config["SECRET_KEY"] = secrets.token_hex(32)
Session(app)
conn_string = os.getenv("AZURE_CONNECTION_STRING")

if not conn_string:
    raise ValueError("AZURE_CONNECTION_STRING is not set.")

blob_service_client = BlobServiceClient.from_connection_string(
    conn_string
)

REPORT_FILE = "profile_report.html"


@app.route('/get_uuid', methods=['POST'])
def get_uuid():
    try:
        data = request.get_json()
        uuid = data.get('uuid')

        if not uuid:
            return jsonify({"error": "No UUID provided"}), 400

        session['uuid'] = uuid
        return jsonify({"message": "UUID successfully stored.", "uuid": uuid}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/<uuid>/get_metadata', methods=['GET'])
def get_metadata(uuid):
    blob_client = blob_service_client.get_blob_client(
            container="bronze",
            blob=f"/Users/{uuid}/input.csv"
        )
    blob_data = blob_client.download_blob().readall()
    df = pd.read_csv(io.BytesIO(blob_data))

    try:
        metadata = {
            "columns": [{"name": col, "dtype": str(df[col].dtype)} for col in df.columns],
            "missing_values": df.isnull().sum().to_dict(),
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "summary_statistics": df.describe().to_dict(),
        }
        return jsonify(metadata), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_unique_id_column', methods=['POST'])
def get_unique_id():
    try:
        data = request.get_json()
        unique_id_column = data.get('unique_id_column')

        if not unique_id_column:
            return jsonify({"error": "No unique_id_column provided"}), 400

        session['unique_id_column'] = unique_id_column
        return jsonify({"message": "Unique ID column successfully stored.", "unique_id_column": unique_id_column}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/<uuid>/generate-profile", methods=["POST"])
def generate_profile(uuid):
    if not uuid:
        return jsonify({"error": "UUID not found. Call /get_uuid first."}), 400

    try:
        
        blob_client = blob_service_client.get_blob_client(
            container="bronze",
            blob=f"/Users/{uuid}/input.csv"
        )
        blob_data = blob_client.download_blob().readall()
        df = pd.read_csv(io.BytesIO(blob_data))
        
        profile = ProfileReport(df, explorative=True)
        profile.to_file(REPORT_FILE)

        return send_from_directory(REPORT_FILE, mimetype="text/html"), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


DATABRICKS_INSTANCE = os.getenv('DATABRICKS_INSTANCE')
DATABRICKS_TOKEN = os.getenv('DATABRICKS_TOKEN')
CLUSTER_ID = os.getenv('CLUSTER_ID')


SESSION_URL = f"https://{DATABRICKS_INSTANCE}/api/1.2/contexts/create"
COMMAND_URL = f"https://{DATABRICKS_INSTANCE}/api/1.2/commands/execute"
STATUS_URL = f"https://{DATABRICKS_INSTANCE}/api/1.2/commands/status"

HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json"
}


@app.route("/<uuid>/run-databricks", methods=["POST"])
def run_databricks(uuid):
    session_payload = {"language": "python", "clusterId": CLUSTER_ID}
    session_response = requests.post(SESSION_URL, headers=HEADERS, json=session_payload)
    session_id = session_response.json().get("id")

    if not session_id:
        return jsonify({"error": "Failed to create execution context", "details": session_response.text}), 500
    
    pyspark_code = f"""
    import pandas as pd
    from azure.storage.blob import BlobServiceClient
    import io
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, isnull
    from scipy.stats import zscore
    
    # Initialize Spark session
    spark = SparkSession.builder.appName("BlobToCSV").getOrCreate()
    
    # Azure Blob Storage connection details
    connection_string = {conn_string}
    container_name = "bronze"
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    
    # Function to download a blob from Azure Storage
    def download_blob(blob_name):
        try:
            blob_client = container_client.get_blob_client(blob_name)
            return blob_client.download_blob().readall()
        except Exception as e:
            print(f"Error downloading blob")
            return None
    
    # Function to check if the blob exists in Azure Blob Storage
    def file_exists(blob_name):
        try:
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.get_blob_properties()  # This will raise an error if the blob doesn't exist
            return True
        except Exception as e:
            print(f"Error")
            return False
    
    # Function to list all files in the Azure Blob Storage container
    def list_files_in_container():
        try:
            blob_list = container_client.list_blobs()
            files = [blob.name for blob in blob_list]
            print(f"Files in the container")
            return files
        except Exception as e:
            print(f"Error listing files in the container")
            return []
    
    # Function to remove outliers using Z-score (values with |Z| > 3)
    def remove_outliers(df):
        
        numerical_columns = [column for column in df.columns if df.schema[column].dataType.simpleString() in ['int', 'double', 'float']]
    
        for column in numerical_columns:
            column_values = [row[column] for row in df.select(column).collect() if row[column] is not None]
            if column_values:
                z_scores = zscore(column_values)
                threshold = 3
                filtered_values = [val for val, z in zip(column_values, z_scores) if abs(z) <= threshold]
                df = df.filter(col(column).isin(filtered_values))
    
        return df
    
    # Function to calculate trust score (same as provided in your initial code)
    def calculate_trust_score(df):
        total_rows = df.count()
        if total_rows == 0:
            return 0.0
    
        # **1. Missing Data Percentage**
        missing_data_percentage = sum(df.filter(isnull(col(column))).count() for column in df.columns) / total_rows * 100
    
        # **2. Duplicate Row Percentage**
        duplicate_percentage = (total_rows - df.distinct().count()) / total_rows * 100
    
        # **3. Outlier Percentage (Before Removal)**
        df_cleaned = remove_outliers(df)
        outlier_percentage = (1 - (df_cleaned.count() / total_rows)) * 100
    
        # **4. Invalid Data Percentage (Negative Values)**
        numerical_columns = [column for column in df.columns if df.schema[column].dataType.simpleString() in ['int', 'double', 'float']]
        invalid_percentage = sum(df.filter(col(column) < 0).count() for column in numerical_columns) / total_rows * 100
    
        # **Compute Final Trust Score**
        penalty_percentage = (missing_data_percentage + duplicate_percentage + outlier_percentage + invalid_percentage) / 4
        trust_score = max(0, 100 - penalty_percentage)  # Ensure it doesn't go below 0
    
        return trust_score
    
    # Main function to process the _output.csv file
    def main():
        try:
            # List all files in the container
            files_in_container = list_files_in_container()
        
            # Specify the output file you want to check (update this with the correct file name)
            output_csv_name = f'Users/{uuid}/input.csv' 
    
            # Check if the file exists in the container
            if output_csv_name in files_in_container:
                print(f"File found. Downloading...")
            
                # Download the _output.csv file from Blob Storage
                blob_data = download_blob(output_csv_name)
            
                if blob_data:
                    # Read the CSV data into a pandas DataFrame
                    df = pd.read_csv(io.BytesIO(blob_data))
    
                    # Create a Spark DataFrame from pandas DataFrame
                    spark_df = spark.createDataFrame(df)
    
                    # Calculate the trust score
                    trust_score = calculate_trust_score(spark_df)
    
                    # Print the final trust score
                    print("Trust Score:", trust_score)
                else:
                    print(f"Failed to download .")
            else:
                print(f"File  does not exist in the container.")
    
        except Exception as e:
            print(f"Error occurred:")
    
    # Execute the main function
    if __name__ == "__main__":
        main()
    """
    command_payload = {"language": "python", "clusterId": CLUSTER_ID, "contextId": session_id, "command": pyspark_code}
    command_response = requests.post(COMMAND_URL, headers=HEADERS, json=command_payload)
    command_id = command_response.json().get("id")

    if not command_id:
        return jsonify({"error": "Failed to send command", "details": command_response.text}), 500
    
    return jsonify({"message": "Command sent successfully", "command_id": command_id, "session_id": session_id})


@app.route("/status/<session_id>/<command_id>", methods=["GET"])
def check_status(session_id, command_id):
    while True:
        status_response = requests.get(STATUS_URL, headers=HEADERS, params={"clusterId": CLUSTER_ID, "contextId": session_id, "commandId": command_id})
        status_json = status_response.json()
        state = status_json.get("status")

        if state in ["Finished", "Cancelled", "Error"]:
            return jsonify({"status": state, "output": status_json.get("results", {}).get("data", "No Output")})
        
        time.sleep(3)

if __name__ == "__main__":
    app.run()

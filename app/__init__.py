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

app = Flask(__name__)

app.config["SESSION_TYPE"] = "filesystem"
app.config["SECRET_KEY"] = secrets.token_hex(32)
Session(app)

conn_string = os.getenv("AZURE_CONNECTION_STRING")

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
    

@app.route('/get_metadata', methods=['GET'])
def get_metadata():
    df_json = session.get('df_json')

    if not df_json:
        return jsonify({"error": "No data found in session. Call /generate-profile first."}), 400

    df = pd.read_json(df_json)  
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


@app.route("/generate-profile", methods=["POST"])
def generate_profile():
    uuid = session.get('uuid')
    if not uuid:
        return jsonify({"error": "UUID not found. Call /get_uuid first."}), 400

    try:
        
        blob_client = blob_service_client.get_blob_client(
            container="bronze",
            blob=f"/Users/{uuid}/input.csv"
        )
        blob_data = blob_client.download_blob().readall()
        df = pd.read_csv(io.BytesIO(blob_data))

        
        session['df_json'] = df.to_json()

        
        profile = ProfileReport(df, explorative=True)
        profile.to_file(REPORT_FILE)

        return jsonify({"message": "Profile report generated", "report_url": "/view-report"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/view-report", methods=["GET"])
def view_report():
    if not os.path.exists(REPORT_FILE):
        return jsonify({"error": "Report not found. Please generate it first."}), 404
    
    return send_from_directory(REPORT_FILE, mimetype="text/html")

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


@app.route("/run-databricks", methods=["POST"])
def run_databricks():
    session_payload = {"language": "python", "clusterId": CLUSTER_ID}
    session_response = requests.post(SESSION_URL, headers=HEADERS, json=session_payload)
    session_id = session_response.json().get("id")

    if not session_id:
        return jsonify({"error": "Failed to create execution context", "details": session_response.text}), 500
    
    pyspark_code = """
    df = spark.read.option('header', 'true').csv("/mnt/bronze/hsptl", inferSchema=True)
    display(df)
    df.write.format("delta").mode("overwrite").save("/mnt/silver/hspt_ak")
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
    app.run(debug=True, host="0.0.0.0", port=8000)

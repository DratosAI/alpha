python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
source .env
mlflow server --host $MLFLOW_TRACKING_SERVER_HOST --port $MLFLOW_TRACKING_SERVER_PORT
mlflow deployments start-server --config-path config.yaml --port {port} --host {host} --workers {worker count}

export LITELLM_LOG="ERROR"
export LITELLM_MODE="PRODUCTION"

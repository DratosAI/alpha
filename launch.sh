python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
mlflow server --host 127.0.0.1 --port 8080
mlflow deployments start-server --config-path config.yaml --port {port} --host {host} --workers {worker count}

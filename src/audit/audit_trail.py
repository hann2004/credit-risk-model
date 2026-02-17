import hashlib
import json
from datetime import datetime


class AuditLogger:
    def __init__(self, log_file="audit_log.json"):
        self.log_file = log_file

    def log_decision(
        self, customer_id, features, prediction, probability, model_version
    ):
        """
        Create immutable audit record
        """
        timestamp = datetime.utcnow().isoformat()
        record_string = f"{timestamp}{customer_id}{prediction}{model_version}"
        record_hash = hashlib.sha256(record_string.encode()).hexdigest()
        audit_record = {
            "timestamp": timestamp,
            "customer_id": customer_id,
            "features_hash": hashlib.sha256(
                json.dumps(features, sort_keys=True).encode()
            ).hexdigest(),
            "prediction": int(prediction),
            "probability": float(probability),
            "model_version": model_version,
            "record_hash": record_hash,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(audit_record) + "\n")

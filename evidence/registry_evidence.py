"""Step 2 evidence: registered versions, aliases, descriptions and tags."""

import json
from pathlib import Path

from mlflow import MlflowClient

OUTPUT_JSON = Path(__file__).resolve().parent / "02_registry.json"
MODEL_NAME = "loan-default"
client = MlflowClient("http://127.0.0.1:5000")

registered = client.get_registered_model(MODEL_NAME)
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
versions = sorted(versions, key=lambda version: int(version.version))

records = []
for version in versions:
    records.append(
        {
            "version": str(version.version),
            "run_id": version.run_id,
            "status": version.status,
            "description": version.description,
            "tags": dict(version.tags),
            "aliases": sorted(
                alias
                for alias in ("champion", "challenger")
                if str(client.get_model_version_by_alias(MODEL_NAME, alias).version) == str(version.version)
            ),
        }
    )

alias_records = {}
for alias in ("champion", "challenger"):
    version = client.get_model_version_by_alias(MODEL_NAME, alias)
    alias_records[alias] = {
        "version": str(version.version),
        "run_id": version.run_id,
        "description": version.description,
        "tags": dict(version.tags),
    }

evidence = {
    "registered_model": registered.name,
    "registered_model_description": registered.description,
    "versions": records,
    "aliases": alias_records,
    "descriptions_present": all(record["description"] for record in records),
    "aliases_distinct": alias_records["champion"]["version"] != alias_records["challenger"]["version"],
}
OUTPUT_JSON.write_text(json.dumps(evidence, indent=2), encoding="utf-8")

print(f"Registered model: {registered.name}")
print(f"Versions        : {len(records)}")
print(f"Aliases distinct: {evidence['aliases_distinct']}")
print(f"Descriptions    : {'all present' if evidence['descriptions_present'] else 'MISSING'}")
print("\nAliases")
for alias, record in alias_records.items():
    print(f"  {alias:<10} -> v{record['version']}  run={record['run_id']}")
    print(f"             {record['description']}")
print("\nVersions")
for record in records:
    print(f"  v{record['version']}  aliases={record['aliases']}  run={record['run_id']}")
    print(f"       tags={record['tags']}")
    print(f"       description={record['description']}")
print(f"\nEvidence: {OUTPUT_JSON}")
raise SystemExit(0 if evidence["descriptions_present"] and evidence["aliases_distinct"] else 1)

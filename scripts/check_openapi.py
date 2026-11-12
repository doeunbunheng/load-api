"""Compare the live /openapi.json with the Module 2 contract.

The contract comparison is by HTTP method and path, as required by the
capstone. Request and response field checks are reported as well. The two
Module 6 additions (the X-API-Key security scheme and the extra health fields)
are listed as intentional differences.
"""

import argparse
import json
import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "spec" / "module2_contract.json"


def _schema_properties(document: dict, operation: dict, request: bool) -> list[str]:
    if request:
        schema = (
            operation.get("requestBody", {})
            .get("content", {})
            .get("application/json", {})
            .get("schema", {})
        )
    else:
        schema = (
            operation.get("responses", {})
            .get("200", {})
            .get("content", {})
            .get("application/json", {})
            .get("schema", {})
        )
    if "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[-1]
        schema = document.get("components", {}).get("schemas", {}).get(name, {})
    return sorted(schema.get("properties", {}).keys())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("LIVE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    response = requests.get(f"{args.url.rstrip('/')}/openapi.json", timeout=30)
    response.raise_for_status()
    document = response.json()

    expected = {
        path: set(operations)
        for path, operations in spec["paths"].items()
    }
    actual = {
        path: set(operations)
        for path, operations in document.get("paths", {}).items()
    }
    missing_paths = sorted(set(expected) - set(actual))
    extra_paths = sorted(set(actual) - set(expected))
    method_differences = {
        path: {
            "expected": sorted(expected[path]),
            "actual": sorted(actual.get(path, set())),
        }
        for path in expected
        if path in actual and expected[path] != actual[path]
    }

    field_differences = []
    for path, operations in spec["paths"].items():
        for method, contract in operations.items():
            operation = document.get("paths", {}).get(path, {}).get(method)
            if not operation:
                continue
            for request in (True, False):
                key = "request_fields" if request else "response_fields"
                if key not in contract:
                    continue
                live_fields = _schema_properties(document, operation, request=request)
                expected_fields = sorted(contract[key])
                missing = sorted(set(expected_fields) - set(live_fields))
                extra = sorted(set(live_fields) - set(expected_fields))
                if missing or extra:
                    field_differences.append(
                        {
                            "path": path,
                            "method": method,
                            "kind": key,
                            "missing": missing,
                            "extra": extra,
                        }
                    )

    method_path_match = not missing_paths and not extra_paths and not method_differences
    missing_required_fields = [
        difference
        for difference in field_differences
        if difference["missing"]
    ]
    passed = method_path_match and not missing_required_fields

    security = document.get("components", {}).get("securitySchemes", {})
    evidence = {
        "live_url": args.url,
        "contract": str(args.spec),
        "expected_paths": {path: sorted(methods) for path, methods in expected.items()},
        "actual_paths": {path: sorted(methods) for path, methods in actual.items()},
        "missing_paths": missing_paths,
        "extra_paths": extra_paths,
        "method_differences": method_differences,
        "field_differences": field_differences,
        "security_schemes": security,
        "intentional_module6_differences": spec.get("intentional_module6_differences", []),
        "passed": passed,
    }

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    print(f"Live OpenAPI : {args.url.rstrip('/')}/openapi.json")
    print(f"Contract     : {args.spec}")
    print(f"Methods/paths match: {method_path_match}")
    if missing_paths:
        print(f"  Missing paths: {missing_paths}")
    if extra_paths:
        print(f"  Extra paths  : {extra_paths}")
    if method_differences:
        print(f"  Method diffs : {json.dumps(method_differences, indent=2)}")
    for difference in field_differences:
        print(
            f"  {difference['method'].upper()} {difference['path']} {difference['kind']}: "
            f"missing={difference['missing']} extra={difference['extra']}"
        )
    print(f"Security schemes: {sorted(security)}")
    print(f"Overall      : {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

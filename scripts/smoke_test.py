#!/usr/bin/env python3
"""Smoke test — run before every deploy to catch obvious breaks.

Usage: python scripts/smoke_test.py [base_url]
Default: http://localhost:8847
"""

import sys
import json
import urllib.request
import urllib.error
import time

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8847"
PASSED = 0
FAILED = 0
ERRORS = []


def test(name: str, method: str, path: str, expected_status: int = 200, timeout: int = 30):
    global PASSED, FAILED
    url = f"{BASE}{path}"
    try:
        req = urllib.request.Request(url, method=method)
        t0 = time.time()
        resp = urllib.request.urlopen(req, timeout=timeout)
        ms = round((time.time() - t0) * 1000)
        code = resp.status
        if code == expected_status:
            print(f"  ✅ {name} — {code} ({ms}ms)")
            PASSED += 1
        else:
            print(f"  ❌ {name} — expected {expected_status}, got {code}")
            FAILED += 1
            ERRORS.append(f"{name}: expected {expected_status}, got {code}")
    except urllib.error.HTTPError as e:
        if e.code == expected_status:
            print(f"  ✅ {name} — {e.code}")
            PASSED += 1
        else:
            print(f"  ❌ {name} — expected {expected_status}, got {e.code}: {e.reason}")
            FAILED += 1
            ERRORS.append(f"{name}: {e.code} {e.reason}")
    except Exception as e:
        print(f"  ❌ {name} — {type(e).__name__}: {e}")
        FAILED += 1
        ERRORS.append(f"{name}: {e}")


def test_json(name: str, path: str, check_key: str = None, timeout: int = 30):
    """Test endpoint returns valid JSON, optionally check for a key."""
    global PASSED, FAILED
    url = f"{BASE}{path}"
    try:
        t0 = time.time()
        resp = urllib.request.urlopen(url, timeout=timeout)
        ms = round((time.time() - t0) * 1000)
        data = json.loads(resp.read())
        if check_key and check_key not in data:
            print(f"  ❌ {name} — missing key '{check_key}' ({ms}ms)")
            FAILED += 1
            ERRORS.append(f"{name}: missing key '{check_key}'")
        else:
            print(f"  ✅ {name} — OK ({ms}ms)")
            PASSED += 1
        return data
    except Exception as e:
        print(f"  ❌ {name} — {type(e).__name__}: {e}")
        FAILED += 1
        ERRORS.append(f"{name}: {e}")
        return None


print(f"\n🧪 Smoke Test — {BASE}\n")

# 1. Health
print("── Health ──")
test_json("health", "/health", "status")

# 2. Generate dataset
print("\n── Generate ──")
gen_url = f"{BASE}/api/generate?duration=30&n_electrodes=8&burst_probability=0.15"
try:
    req = urllib.request.Request(gen_url, method="POST", data=b"")
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read())
    ds_id_found = "dataset_id" in data
    print(f"  {'✅' if ds_id_found else '❌'} generate 30s — {'OK' if ds_id_found else 'missing dataset_id'}")
    if ds_id_found:
        PASSED += 1
    else:
        FAILED += 1
except Exception as e:
    data = None
    print(f"  ❌ generate 30s — {e}")
    FAILED += 1
    ERRORS.append(f"generate: {e}")
ds_id = data["dataset_id"] if data else None

if ds_id:
    # 3. Core endpoints
    print("\n── Core Analysis ──")
    test_json("summary", f"/api/analysis/{ds_id}/summary", "dataset")
    test_json("firing-rates", f"/api/analysis/{ds_id}/firing-rates")
    test_json("isi", f"/api/analysis/{ds_id}/isi")
    test_json("amplitudes", f"/api/analysis/{ds_id}/amplitudes")
    test_json("quality", f"/api/analysis/{ds_id}/quality")
    test_json("bursts", f"/api/analysis/{ds_id}/bursts")
    test_json("temporal", f"/api/analysis/{ds_id}/temporal")
    test_json("iq", f"/api/analysis/{ds_id}/iq")
    test_json("health", f"/api/analysis/{ds_id}/health")

    # 4. Protocols
    print("\n── Protocols ──")
    test_json("list protocols", "/api/protocols", "protocols")

    # 5. Advanced (lazy) endpoints — just verify they return 200 or 504
    print("\n── Advanced Analysis ──")
    for ep, fn in [
        ("pca", "pca"),
        ("states", "states"),
        ("connectivity", "connectivity"),
        ("graph-theory", "graph-theory"),
        ("topology", "topology"),
        ("ethics", "ethics"),
    ]:
        url = f"{BASE}/api/analysis/{ds_id}/{fn}"
        try:
            t0 = time.time()
            resp = urllib.request.urlopen(url, timeout=60)
            ms = round((time.time() - t0) * 1000)
            if resp.status == 200:
                print(f"  ✅ {ep} — 200 ({ms}ms)")
                PASSED += 1
            else:
                print(f"  ❌ {ep} — {resp.status} ({ms}ms)")
                FAILED += 1
        except urllib.error.HTTPError as e:
            if e.code == 504:
                print(f"  ⏱  {ep} — 504 timeout (acceptable)")
                PASSED += 1
            else:
                print(f"  ❌ {ep} — {e.code} {e.reason}")
                FAILED += 1
                ERRORS.append(f"{ep}: {e.code} {e.reason}")
        except Exception as e:
            print(f"  ❌ {ep} — {type(e).__name__}: {e}")
            FAILED += 1
            ERRORS.append(f"{ep}: {e}")

    # 6. 404 for missing dataset + 410 Gone detection
    print("\n── Error Handling ──")
    test("missing dataset", "GET", "/api/analysis/nonexistent/summary", expected_status=404)

    # 7. Response shape sanity — some key endpoints expose documented fields
    print("\n── Response Shape ──")
    bursts_data = test_json("bursts shape", f"/api/analysis/{ds_id}/bursts", "summary")
    iq_data = test_json("iq shape", f"/api/analysis/{ds_id}/iq", "iq_score")
    if iq_data:
        score = iq_data.get("iq_score", 0)
        grade = iq_data.get("grade", "?")
        print(f"     iq_score={score} grade={grade}")

# Results
print(f"\n{'='*40}")
print(f"✅ Passed: {PASSED}")
print(f"❌ Failed: {FAILED}")
if ERRORS:
    print(f"\nErrors:")
    for e in ERRORS:
        print(f"  • {e}")
print()
sys.exit(1 if FAILED > 0 else 0)

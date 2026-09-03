import time
import csv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# 3-Way Cloud Provider Configuration
CONFIG = {
    "Google_Cloud_Run": {
        "url": "https://cold-start-test-513782382242.asia-south1.run.app",
        "file": "gcp_latency_data.csv"
    },
    "AWS_Lambda": {
        # Confirmed against your working curl test: the version pasted in
        # for "optimization" had an extra "x" ("ighxhxm7...") that doesn't
        # match this ("ighhxm7...") -- that typo would turn every AWS ping
        # into a dead DNS lookup instead of a real request.
        "url": "https://ighxhxm7nqfvwdigpcpm526yia0mtnxi.lambda-url.eu-north-1.on.aws/",
        "file": "aws_lambda_latency_data.csv"
    },
    "Azure_Functions": {
        "url": "https://azure-coldstart-test-cxf6chcdc5brgjcz.indiasouthcentral-01.azurewebsites.net/api/HttpTrigger1",
        "file": "azure_latency_data.csv"
    }
}


def log_to_csv(filename, phase, latency_ms, status_code):
    """Logs individual ping latency data into the provider's dedicated CSV file."""
    try:
        with open(filename, "a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["Timestamp", "Phase", "Latency_ms", "Status_Code"])
            writer.writerow([datetime.now().isoformat(), phase, latency_ms, status_code])
    except Exception as e:
        print(f"❌ File Write Error ({filename}): {e}")


def ping_one(url):
    """Pings a single URL. Returns (elapsed_ms, status) whether the request
    succeeded or not, so a timeout/connection error still produces a
    loggable result instead of vanishing silently (previously a failed
    request was only printed, never written to the CSV -- indistinguishable
    later from "we never tried")."""
    start = time.perf_counter()
    try:
        response = requests.get(url, timeout=30)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return elapsed_ms, response.status_code
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        return elapsed_ms, type(e).__name__


def ping_all(phase_name):
    """Pings GCP, AWS, and Azure concurrently and logs results. (Previously
    this looped through them one at a time despite the docstring saying
    "concurrently" -- each phase took the sum of all three latencies
    instead of the slowest one, and later providers were pinged noticeably
    later than earlier ones within the same "simultaneous" phase.)"""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] --- {phase_name} ---")
    with ThreadPoolExecutor(max_workers=len(CONFIG)) as pool:
        futures = {
            pool.submit(ping_one, details["url"]): (provider, details)
            for provider, details in CONFIG.items()
        }
        for future in as_completed(futures):
            provider, details = futures[future]
            elapsed_ms, status = future.result()
            log_to_csv(details["file"], phase_name, elapsed_ms, status)
            icon = "✅" if isinstance(status, int) else "❌"
            print(f"{icon} {provider}: {elapsed_ms} ms (Status: {status})")


print("🚀 Starting 3-Way Serverless Latency Benchmark (GCP vs. AWS vs. Azure)")

# 3 Full Experimental Cycles (Cold Start -> 3 Warm Pings -> 35-minute Cooldown)
TOTAL_CYCLES = 9

for cycle in range(1, TOTAL_CYCLES + 1):
    print(f"\n{'='*20} STARTING CYCLE {cycle}/{TOTAL_CYCLES} {'='*20}")
    # 1. Cold Start Invocations
    ping_all(f"Cycle_{cycle}_Cold_Start")
    # 2. Warm State Invocations
    for i in range(1, 4):
        print(f"Waiting 5 seconds before Warm Ping {i}...")
        time.sleep(5)
        ping_all(f"Cycle_{cycle}_Warm_{i}")
    # 3. Cooldown Phase (Allows all providers to scale down to zero)
    if cycle < TOTAL_CYCLES:
        print("\n⏳ Cooldown in progress: waiting 35 minutes to trigger scale-to-zero...")
        for minutes_left in range(35, 0, -1):
            print(f"{minutes_left} minutes remaining...")
            time.sleep(60)

print("\n🎉 Benchmark Complete! Data recorded in:")
print("   - gcp_latency_data.csv")
print("   - aws_lambda_latency_data.csv")
print("   - azure_latency_data.csv")

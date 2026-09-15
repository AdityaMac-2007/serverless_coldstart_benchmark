# Empirical Analysis of Cold Starts in Serverless Computing

This repository contains the replication dataset, telemetry runner logs, and automated visualization pipeline for the research paper:

> **"Empirical Analysis of Cold Starts in Serverless Computing Across Major Cloud Providers"**  
> *Dr. Shalu Singh, Aditya Chand, Swayampurna Mandal, Tanisha Singla*  
> Department of Computer Science and Technology, Manav Rachna University

---

## Repository Structure

* `aws_lambda_latency_data.csv`: Primary experiment telemetry for AWS Lambda (`eu-north-1`, 83 cycles / 332 calls).
* `gcp_latency_data.csv`: Primary experiment telemetry for Google Cloud Run (`asia-south1`, 83 cycles / 332 calls).
* `azure_latency_data.csv`: Primary experiment telemetry for Azure Functions (`indiasouthcentral`, 83 cycles / 332 calls).
* `aws_mumbai_backup_latency_data.csv`: Domestic replication telemetry for AWS Lambda (`ap-south-1`, 99 cycles / 396 calls).
* `gcp_backup_latency_data.csv`: Domestic replication telemetry for Google Cloud Run (`asia-south1`, 99 cycles / 396 calls).
* `azure_backup_latency_data.csv`: Domestic replication telemetry for Azure Functions (`indiasouthcentral`, 99 cycles / 396 calls).
* `Graph.py`: Complete visualization pipeline generating publication-ready ECDF, timeseries, and geographic invariance figures.
* `figures_fixed/`: Generated output figures in 300 DPI PNG and vector PDF formats.

---

## Reproduction Instructions

### 1. Requirements
Ensure Python 3.9+ is installed along with the following standard libraries:
```bash
pip install pandas numpy matplotlib

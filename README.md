# 🚀 AI-Powered API Monitoring & Anomaly Detection System

An intelligent monitoring and anomaly detection platform designed for large-scale, distributed API environments across on-premise, cloud, and multi-cloud infrastructures.

This system combines observability tooling with machine learning to provide real-time monitoring, predictive analytics, automated alerts, and AI-driven root cause analysis.

---

## 🧠 Problem It Solves

Modern distributed API systems often struggle with:

- Latency spikes
- Sudden error rate increases
- Hidden service dependencies
- Delayed incident detection
- Manual root cause analysis

This platform delivers:

- Intelligent anomaly detection
- Predictive failure forecasting
- Automated incident response
- AI-generated Root Cause Analysis (RCA) reports

---

## 🚀 Key Features

### 🤖 AI Alert & Automation System

- Automated phone alerts triggered by logs, metrics, and traces  
- Voice-enabled incident reporting  
- Automated recovery workflows  
- On-demand AI-generated RCA reports  

---

### 🔍 Intelligent Root Cause Analysis

- Correlates logs, metrics, and traces  
- Time-series forecasting using Prophet  
- Real-time anomaly detection via Isolation Forest  
- Automated RCA report generation  

---

### 📊 Real-Time Observability Stack

- Grafana dashboards for visualization  
- Loki for centralized log aggregation  
- Tempo for distributed tracing  
- Mimir for long-term metrics storage  

---

### 🔥 Advanced Anomaly Detection

- Isolation Forest (unsupervised ML)  
- Detects latency, error rate, and traffic anomalies  
- Environment-aware alert thresholds  
- Real-time anomaly scoring  

---

### 📈 Predictive Analytics

- Prophet-based time-series forecasting  
- Proactive alerting before failures occur  
- Capacity planning insights  
- Trend analysis and resource optimization  

---

### 🧭 End-to-End API Tracking

- Complete API request journey tracing  
- Cross-service correlation  
- Dependency mapping  
- Bottleneck identification  

---

## 🏗️ System Architecture


**Observability Stack**
- Loki (Logs)
- Tempo (Traces)
- Mimir (Metrics)

**AI/ML Engine**
- Prophet Forecasting
- Isolation Forest Anomaly Detection

**Alert System**
- Phone Alerts
- RCA Reports
- Automated Recovery

---

## 🛠️ Core Components

- Node.js Express API (Telemetry Generator)
- OpenTelemetry SDK & Collector
- Grafana (Visualization)
- Loki (Logs)
- Tempo (Tracing)
- Mimir (Metrics)
- Prophet (Forecasting)
- Isolation Forest (Anomaly Detection)
- AI Alert System (Automation & RCA)

---

## 📋 Prerequisites

- Node.js (v16+)
- Docker & Docker Compose
- Python (v3.8+)
- Twilio Account (for phone alerts)
- Gemini API Key (for RCA generation)

---

## 🚀 Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone <repository-url>
cd ai-api-monitoring-system

2️⃣ Install Dependencies

npm install
pip install -r requirements.txt

3️⃣ Configure Environment Variables

Create .env file:
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_PHONE_NUMBER=your_number
GEMINI_API_KEY=your_key
ALERT_PHONE_NUMBERS=+1234567890
NODE_ENV=production
API_PORT=8080

4️⃣ Start Infrastructure
docker-compose up -d
docker-compose ps

5️⃣ Initialize Observability
sleep 60
./scripts/setup-dashboards.sh
./scripts/setup-datasources.sh
6️⃣ Start Services

npm start

python ai/anomaly_detector.py &
python ai/prophet_forecaster.py &
python ai/alert_system.py &
python ai/rca_engine.py &

🧪 Testing the System
API Endpoints

curl http://localhost:8080/rolldice
curl http://localhost:8080/health
curl http://localhost:8080/error
curl http://localhost:8080/heavy
curl http://localhost:8080/db-query

Load Testing
artillery run load-test.yml
python scripts/load_test.py --duration 300 --rps 50

📊 Access Dashboards

Grafana:
http://localhost:3000
Default: admin / admin

Preconfigured dashboards include:

API Overview

Anomaly Detection

Predictive Analytics

Distributed Tracing

Infrastructure Monitoring

📈 Key Metrics Monitored

Response Time (P50 / P95 / P99)

Error Rate (4xx / 5xx)

Request Throughput (RPS)

Anomaly Scores


Forecast Accuracy

🤖 AI/ML Capabilities
Prophet Forecasting

Traffic prediction

Latency forecasting

Capacity planning

Isolation Forest

Detects latency anomalies

Identifies unusual traffic patterns

Flags error spikes

Cross-service anomaly correlation


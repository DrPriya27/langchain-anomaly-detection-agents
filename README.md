# Anomaly Detection Agent

## Overview

The Anomaly Detection Agent is an advanced AI-powered tool designed to detect anomalies in marketing metrics data. It uses statistical methods to identify unusual patterns such as spend spikes, performance drops, and ratio anomalies in marketing campaigns. Built with data science best practices and integrated with LangChain for natural language processing, this agent helps marketing teams quickly identify issues and opportunities in their campaign performance data.

Impressions refer to the number of times your ad was displayed to a user.

Clicks measure the number of times your ad was clicked on.

Conversion rate is a percentage of users who completed a desired action.

Cost Per Acquisition (CPA) refers to the total amount it costs to get a user from the beginning to the end of the sales funnel.

Click-through rate (CTR) is a percentage of how many times your ad was clicked on out of the total times it was displayed

## Features

- **Multi-metric Anomaly Detection**: Identifies anomalies across various marketing metrics:
  - Spend spikes (unusual increases in campaign spend)
  - Impression and click drops (unexpected decreases in performance)
  - Conversion rate anomalies (unusual changes in effectiveness)
  - CTR (Click-Through Rate) anomalies
  - CPA (Cost Per Acquisition) anomalies

- **Campaign-specific Analysis**: Filter anomaly detection by specific campaigns or analyze across all campaigns

- **Data Visualization**: Generate visual representations of metrics with anomalies highlighted for easy identification

- **Statistical Analysis**: Uses moving window statistics and z-scores to detect anomalies with configurable sensitivity

- **Summary Statistics**: Provides summary statistics for marketing metrics including mean, median, standard deviation, and totals

- **Natural Language Interface**: Process user queries in natural language to run specific anomaly detection operations

## Components

The Anomaly Detection Agent consists of several key components:

### 1. AnomalyDetector Class

The core class that implements anomaly detection algorithms with methods for:

- `detect_spend_spikes()`: Identifies unusually high spending periods
- `detect_metric_drops()`: Identifies significant drops in any performance metric
- `detect_ratio_anomalies()`: Analyzes derived metrics like CTR and conversion rate
- `visualize_anomalies()`: Creates visualizations of metrics with anomalies highlighted
- `get_summary_statistics()`: Calculates descriptive statistics for all metrics

### 2. LangChain Integration

Integrates with LangChain and Azure OpenAI to provide:

- Natural language query processing
- AI-powered analysis of anomaly results
- Tool-based agent for handling complex queries

### 3. Command Line Interface

Provides direct interaction through a command-line interface for:

- Querying specific anomalies
- Generating visualizations
- Getting metric summaries

## Usage

### Basic Usage

```python
from anomaly_detection_agent import AnomalyDetector

# Initialize detector with configuration
detector = AnomalyDetector(lookback_days=7, threshold_multiplier=2.0)

# Load data
detector.load_data("marketing_metrics.csv")

# Detect spend anomalies
spend_anomalies = detector.detect_spend_spikes(column='cost')
print(spend_anomalies)

# Visualize the anomalies
detector.visualize_anomalies('spend', spend_anomalies, output_file="spend_anomalies.png")


from anomaly_detection_agent import AnomalyDetector

# Initialize detector with configuration
detector = AnomalyDetector(lookback_days=7, threshold_multiplier=2.0)

# Load data
detector.load_data("marketing_metrics.csv")

# Detect spend anomalies
impressions_anomalies = detector.detect_metric_drops(column='impressions')
print(impressions_anomalies)

# Visualize the anomalies
detector.visualize_anomalies('spend', spend_anomalies, output_file="spend_anomalies.png")


```

### Using the Agent Interface

```python
from anomaly_detection_agent import agent

# Run a query through the agent
response = agent.run("Check for any spend spikes in Campaign A and visualize the results")
print(response)
```

### Direct Function Calls

```python
from anomaly_detection_agent import run_direct_detection

# Run detection with visualization
result = run_direct_detection("Check for CTR anomalies in all campaigns", generate_visuals=True)
```

## Command Line Usage

```bash
# Run the agent with default settings
python anomaly_detection_agent.py

# Run with specific query
python -c "from anomaly_detection_agent import agent; print(agent.run('Check for unusual CPA increases in the last week'))"
```

## Configuration Options

The anomaly detector can be configured with several parameters:

- **lookback_days**: Number of days to use as baseline for anomaly detection (default: 7)
- **threshold_multiplier**: Sensitivity of anomaly detection; higher values detect only more extreme anomalies (default: 2.0)
- **alert_threshold**: Optional absolute threshold for alerts
- **custom_visualization**: Configure visualization appearance

## Data Format

The agent expects a CSV file with the following columns:

- **date**: Date of the metrics (YYYY-MM-DD format)
- **campaign** (optional): Campaign name/identifier
- **spend**: Daily spend amount
- **impressions**: Number of impressions
- **clicks**: Number of clicks
- **conversions**: Number of conversions
- **cpa**: Cost per acquisition

Additional metrics can be added and will be automatically included in the analysis.

## Examples

### Example 1: Detecting Spend Spikes

```python
detector = AnomalyDetector()
detector.load_data("marketing_metrics.csv")
anomalies = detector.detect_spend_spikes(column='spend', campaign='Campaign A')

# Sample output:
# [
#   {
#     "date": "2025-06-12",
#     "campaign": "Campaign A",
#     "metric": "spend",
#     "value": 1850.30,
#     "baseline_mean": 1198.64,
#     "threshold": 1328.95,
#     "percent_increase": 54.36,
#     "z_score": 5.12,
#     "anomaly_type": "spike"
#   }
# ]
```

### Example 2: Visualizing Click Drops

```python
detector = AnomalyDetector()
detector.load_data("marketing_metrics.csv")
anomalies = detector.detect_metric_drops(column='clicks')
detector.visualize_anomalies('clicks', anomalies, output_file="click_anomalies.png")
```

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install pandas numpy matplotlib langchain langchain-openai
```

3. Configure Azure OpenAI credentials in environment variables:
   - Set `AZURE_OPENAI_ENDPOINT`
   - Set `AZURE_OPENAI_API_KEY`

## Dependencies

- Python 3.8+
- pandas
- numpy
- matplotlib
- langchain
- langchain-openai
- Azure OpenAI API access (optional)

## Error Handling

The agent includes robust error handling:

- Fallback to direct detection if agent initialization fails
- Handling of missing data and division by zero
- JSON encoding for complex NumPy types
- Comprehensive error reporting

## Limitations

- Requires sufficient historical data (at least lookback_days + 1) for baseline calculation
- Simple statistical methods may not capture complex seasonality patterns
- Visualization requires matplotlib support in the environment

## Future Enhancements

- Advanced anomaly detection using machine learning models
- Real-time monitoring and alerting functionality
- Integration with marketing platforms via APIs
- Support for more complex seasonality patterns
- Interactive dashboard for anomaly exploration

## License

[Include license information here]

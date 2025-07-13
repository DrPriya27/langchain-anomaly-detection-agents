import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from langchain.agents import initialize_agent, Tool
from langchain_openai import AzureChatOpenAI
import json

# Add a custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class AnomalyDetector:
    """Class to detect anomalies in marketing metrics data"""
    
    def __init__(self, data=None, lookback_days=7, threshold_multiplier=2.0):
        """
        Initialize the anomaly detector
        
        Args:
            data: pandas DataFrame with marketing metrics
            lookback_days: number of days to look back for establishing baseline
            threshold_multiplier: multiplier for standard deviation to determine threshold
        """
        self.data = data
        self.lookback_days = lookback_days
        self.threshold_multiplier = threshold_multiplier
        
    def load_data(self, file_path):
        """Load data from a CSV file"""
        try:
            self.data = pd.read_csv(file_path)
            # Convert date column to datetime
            if 'date' in self.data.columns:
                self.data['date'] = pd.to_datetime(self.data['date'])
            print(f"Data loaded successfully with {len(self.data)} rows")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def detect_spend_spikes(self, column='spend', alert_threshold=None, campaign=None):
        """
        Detect spikes in spend compared to recent history
        
        Args:
            column: name of the spend column to analyze
            alert_threshold: optional custom threshold override
            campaign: filter for specific campaign, None for all
        
        Returns:
            List of anomalies with dates and values
        """
        if self.data is None or column not in self.data.columns:
            return {"error": f"No data loaded or column {column} not found"}
        
        # Filter by campaign if specified
        data = self.data.copy()
        if campaign and 'campaign' in data.columns:
            data = data[data['campaign'] == campaign]
            if len(data) == 0:
                return {"error": f"No data found for campaign: {campaign}"}
        
        # Sort data by date to ensure proper order
        data = data.sort_values('date')
        
        anomalies = []
        
        # We need at least lookback_days + 1 days of data for detection
        min_required_days = self.lookback_days + 1
        
        if len(data) < min_required_days:
            return {"error": f"Not enough data, need at least {min_required_days} days"}
        
        # For each campaign if campaign column exists and no specific campaign is requested
        if 'campaign' in data.columns and campaign is None:
            for campaign_name in data['campaign'].unique():
                campaign_data = data[data['campaign'] == campaign_name].sort_values('date')
                
                if len(campaign_data) < min_required_days:
                    continue  # Skip if not enough data for this campaign
                
                campaign_anomalies = self._detect_anomalies_in_series(
                    campaign_data, column, 'spike', alert_threshold, campaign_name)
                anomalies.extend(campaign_anomalies)
        else:
            # Detect anomalies in the entire dataset or filtered campaign data
            anomalies = self._detect_anomalies_in_series(
                data, column, 'spike', alert_threshold, campaign)
        
        return anomalies
    
    def detect_metric_drops(self, column='impressions', drop_threshold=None, campaign=None):
        """
        Detect significant drops in any metric compared to recent history
        
        Args:
            column: name of the metric column to analyze
            drop_threshold: optional custom threshold override
            campaign: filter for specific campaign, None for all
        
        Returns:
            List of anomalies with dates and values
        """
        if self.data is None or column not in self.data.columns:
            return {"error": f"No data loaded or column {column} not found"}
        
        # Filter by campaign if specified
        data = self.data.copy()
        if campaign and 'campaign' in data.columns:
            data = data[data['campaign'] == campaign]
            if len(data) == 0:
                return {"error": f"No data found for campaign: {campaign}"}
        
        # Sort data by date to ensure proper order
        data = data.sort_values('date')
        
        anomalies = []
        
        # We need at least lookback_days + 1 days of data for detection
        min_required_days = self.lookback_days + 1
        
        if len(data) < min_required_days:
            return {"error": f"Not enough data, need at least {min_required_days} days"}
        
        # For each campaign if campaign column exists and no specific campaign is requested
        if 'campaign' in data.columns and campaign is None:
            for campaign_name in data['campaign'].unique():
                campaign_data = data[data['campaign'] == campaign_name].sort_values('date')
                
                if len(campaign_data) < min_required_days:
                    continue  # Skip if not enough data for this campaign
                
                campaign_anomalies = self._detect_anomalies_in_series(
                    campaign_data, column, 'drop', drop_threshold, campaign_name)
                anomalies.extend(campaign_anomalies)
        else:
            # Detect anomalies in the entire dataset or filtered campaign data
            anomalies = self._detect_anomalies_in_series(
                data, column, 'drop', drop_threshold, campaign)
        
        return anomalies
    
    def _detect_anomalies_in_series(self, data, column, detection_type='spike', custom_threshold=None, campaign=None):
        """
        Helper method to detect anomalies in a time series
        
        Args:
            data: DataFrame with sorted time series data
            column: column to analyze
            detection_type: 'spike' or 'drop'
            custom_threshold: override threshold
            campaign: campaign name for identification
        
        Returns:
            List of anomalies
        """
        anomalies = []
        
        # For each day (except first lookback_days)
        for i in range(self.lookback_days, len(data)):
            current_date = data.iloc[i]['date']
            current_value = data.iloc[i][column]
            
            # Get lookback window
            lookback_data = data.iloc[i-self.lookback_days:i][column]
            
            # Calculate baseline statistics
            baseline_mean = lookback_data.mean()
            baseline_std = lookback_data.std()
            
            # Apply padding to avoid division by zero
            if baseline_std == 0:
                baseline_std = 0.001
                
            # Calculate threshold based on detection type
            if detection_type == 'spike':
                threshold = custom_threshold if custom_threshold is not None else baseline_mean + (self.threshold_multiplier * baseline_std)
                anomaly_condition = current_value > threshold
                percent_change = ((current_value - baseline_mean) / baseline_mean) * 100 if baseline_mean > 0 else 0
                z_score = (current_value - baseline_mean) / baseline_std
                change_type = "increase"
            else:  # drop
                threshold = custom_threshold if custom_threshold is not None else baseline_mean - (self.threshold_multiplier * baseline_std)
                anomaly_condition = current_value < threshold
                percent_change = ((baseline_mean - current_value) / baseline_mean) * 100 if baseline_mean > 0 else 0
                z_score = (baseline_mean - current_value) / baseline_std
                change_type = "decrease"
                
            # Check if current value exceeds/falls below threshold
            if anomaly_condition:
                anomaly = {
                    "date": current_date.strftime('%Y-%m-%d'),
                    "campaign": campaign if campaign else data.iloc[i].get('campaign', 'All'),
                    "metric": column,
                    "value": current_value,
                    "baseline_mean": baseline_mean,
                    "threshold": threshold,
                    f"percent_{change_type}": percent_change,
                    "z_score": z_score,
                    "anomaly_type": detection_type
                }
                anomalies.append(anomaly)
        
        return anomalies
    
    def detect_ratio_anomalies(self, numerator='conversions', denominator='clicks', threshold_multiplier=None, campaign=None):
        """
        Detect anomalies in ratios like CTR, conversion rate, etc.
        
        Args:
            numerator: column name for the numerator
            denominator: column name for the denominator
            threshold_multiplier: optional override for threshold calculation
            campaign: filter for specific campaign, None for all
        
        Returns:
            List of anomalies in the ratio with dates and values
        """
        if self.data is None or numerator not in self.data.columns or denominator not in self.data.columns:
            return {"error": f"No data loaded or columns {numerator}/{denominator} not found"}
        
        # Filter by campaign if specified
        data = self.data.copy()
        if campaign and 'campaign' in data.columns:
            data = data[data['campaign'] == campaign]
            if len(data) == 0:
                return {"error": f"No data found for campaign: {campaign}"}
        
        # Calculate the ratio for all data points
        ratio_name = f"{numerator}_{denominator}_ratio"
        data[ratio_name] = data[numerator] / data[denominator].replace(0, 0.001)  # Avoid division by zero
        
        # Use the regular detection methods on the calculated ratio
        mult = threshold_multiplier if threshold_multiplier else self.threshold_multiplier
        
        # Create a temporary detector with the modified data
        temp_detector = AnomalyDetector(data, self.lookback_days, mult)
        
        # Check for both spikes and drops
        spikes = temp_detector.detect_spend_spikes(column=ratio_name, campaign=campaign)
        drops = temp_detector.detect_metric_drops(column=ratio_name, campaign=campaign)
        
        # Combine and return results
        anomalies = []
        if isinstance(spikes, list):
            for anomaly in spikes:
                anomaly["type"] = "spike"
                anomalies.append(anomaly)
        
        if isinstance(drops, list):
            for anomaly in drops:
                anomaly["type"] = "drop"
                anomalies.append(anomaly)
                
        return anomalies
    
    def visualize_anomalies(self, column, anomalies, output_file=None, campaign=None):
        """
        Create a visualization of the anomalies
        
        Args:
            column: column name to visualize
            anomalies: list of anomalies from detection methods
            output_file: optional file path to save the visualization
            campaign: filter for specific campaign, None for all
        """
        if self.data is None or column not in self.data.columns:
            return {"error": f"No data loaded or column {column} not found"}
        
        # Filter by campaign if specified
        data = self.data.copy()
        if campaign and 'campaign' in data.columns:
            data = data[data['campaign'] == campaign]
            if len(data) == 0:
                return {"error": f"No data found for campaign: {campaign}"}
            
        # Sort data
        data = data.sort_values('date')
        
        plt.figure(figsize=(12, 6))
        
        # If we have campaign data but no specific campaign is requested, plot each campaign separately
        if 'campaign' in data.columns and campaign is None:
            for campaign_name in sorted(data['campaign'].unique()):
                campaign_data = data[data['campaign'] == campaign_name]
                plt.plot(campaign_data['date'], campaign_data[column], 
                         marker='o', markersize=3, label=f"{campaign_name} {column}")
        else:
            plt.plot(data['date'], data[column], marker='o', markersize=3, label=column)
        
        # Plot anomalies as red dots with annotations
        plotted_anomalies = 0
        for anomaly in anomalies:
            if isinstance(anomaly, dict) and anomaly.get('metric') == column:
                # Only plot anomalies for the requested campaign or all anomalies if no campaign is specified
                if campaign is None or anomaly.get('campaign') == campaign or anomaly.get('campaign') == 'All':
                    anomaly_date = datetime.strptime(anomaly['date'], '%Y-%m-%d')
                    plt.scatter(anomaly_date, anomaly['value'], color='red', s=100, zorder=5)
                    
                    # Add annotation with percent change
                    if 'percent_increase' in anomaly:
                        percent = anomaly['percent_increase']
                        change_type = 'increase'
                    elif 'percent_decrease' in anomaly:
                        percent = anomaly['percent_decrease']
                        change_type = 'decrease'
                    else:
                        percent = 0
                        change_type = 'change'
                        
                    plt.annotate(f"{percent:.1f}% {change_type}",
                                xy=(anomaly_date, anomaly['value']),
                                xytext=(10, 10),
                                textcoords='offset points',
                                arrowprops=dict(arrowstyle='->', color='red'),
                                fontsize=8,
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
                    plotted_anomalies += 1
        
        # Add red dots to legend only if we plotted anomalies
        if plotted_anomalies > 0:
            plt.scatter([], [], color='red', s=100, label='Anomalies')
            
        campaign_title = f" for {campaign}" if campaign else ""
        plt.title(f"Anomaly Detection: {column.capitalize()}{campaign_title}")
        plt.xlabel("Date")
        plt.ylabel(column.capitalize())
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            return {"message": f"Visualization saved to {output_file}"}
        else:
            plt.show()
            return {"message": "Visualization displayed"}
    
    def get_summary_statistics(self, campaign=None):
        """
        Get summary statistics for the dataset
        
        Args:
            campaign: filter for specific campaign, None for all
        
        Returns:
            Dictionary with summary statistics
        """
        if self.data is None:
            return {"error": "No data loaded"}
            
        data = self.data.copy()
        if campaign and 'campaign' in data.columns:
            data = data[data['campaign'] == campaign]
            if len(data) == 0:
                return {"error": f"No data found for campaign: {campaign}"}
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        summary = {
            "date_range": {
                "start": data['date'].min().strftime('%Y-%m-%d'),
                "end": data['date'].max().strftime('%Y-%m-%d'),
                "days": (data['date'].max() - data['date'].min()).days + 1
            },
            "metrics": {}
        }
        
        for col in numeric_columns:
            summary["metrics"][col] = {
                "mean": data[col].mean(),
                "median": data[col].median(),
                "std": data[col].std(),
                "min": data[col].min(),
                "max": data[col].max(),
                "total": data[col].sum()
            }
            
        return summary


def run_anomaly_detection(query):
    """Function to run anomaly detection based on user query"""
    try:
        # Parse query to understand what type of anomaly to detect
        detector = AnomalyDetector(lookback_days=7, threshold_multiplier=1.8)
        
        # Load sample data
        data_loaded = detector.load_data("marketing_metrics.csv")
        if not data_loaded:
            return json.dumps({"error": "Failed to load marketing metrics data"}, indent=2, cls=NumpyEncoder)
        
        results = {}
        campaign = None
        
        # Extract campaign from query if present
        campaign_keywords = ["campaign a", "campaign b", "campaign_a", "campaign_b"]
        for keyword in campaign_keywords:
            if keyword in query.lower():
                campaign = keyword.replace("_", " ").title()
        
        # Check for visualization requests
        generate_visuals = "visualize" in query.lower() or "chart" in query.lower() or "graph" in query.lower()
        visual_results = {}
        
        # Check for specific metrics
        if "spend" in query.lower() or "cost" in query.lower():
            spend_anomalies = detector.detect_spend_spikes(column='spend', campaign=campaign)
            results["spend_anomalies"] = spend_anomalies
            
            if generate_visuals and isinstance(spend_anomalies, list) and spend_anomalies:
                detector.visualize_anomalies('spend', spend_anomalies, 
                                            output_file="spend_anomalies.png", 
                                            campaign=campaign)
                visual_results["spend_visualization"] = "Generated spend_anomalies.png"
            
        if "impression" in query.lower():
            impression_anomalies = detector.detect_metric_drops(column='impressions', campaign=campaign)
            results["impression_anomalies"] = impression_anomalies
            
            if generate_visuals and isinstance(impression_anomalies, list) and impression_anomalies:
                detector.visualize_anomalies('impressions', impression_anomalies, 
                                            output_file="impression_anomalies.png", 
                                            campaign=campaign)
                visual_results["impressions_visualization"] = "Generated impression_anomalies.png"
            
        if "click" in query.lower():
            click_anomalies = detector.detect_metric_drops(column='clicks', campaign=campaign)
            results["click_anomalies"] = click_anomalies
            
            if generate_visuals and isinstance(click_anomalies, list) and click_anomalies:
                detector.visualize_anomalies('clicks', click_anomalies, 
                                           output_file="click_anomalies.png", 
                                           campaign=campaign)
                visual_results["clicks_visualization"] = "Generated click_anomalies.png"
            
        if "ctr" in query.lower() or "ratio" in query.lower():
            ctr_anomalies = detector.detect_ratio_anomalies(
                numerator='clicks', denominator='impressions', campaign=campaign)
            results["ctr_anomalies"] = ctr_anomalies
            
        if "conversion" in query.lower():
            conv_anomalies = detector.detect_metric_drops(column='conversions', campaign=campaign)
            results["conversion_anomalies"] = conv_anomalies
            
            if generate_visuals and isinstance(conv_anomalies, list) and conv_anomalies:
                detector.visualize_anomalies('conversions', conv_anomalies, 
                                            output_file="conversion_anomalies.png", 
                                            campaign=campaign)
                visual_results["conversions_visualization"] = "Generated conversion_anomalies.png"
                
        if "cpa" in query.lower() or "cost per acquisition" in query.lower():
            cpa_anomalies = detector.detect_spend_spikes(column='cpa', campaign=campaign)
            results["cpa_anomalies"] = cpa_anomalies
            
            if generate_visuals and isinstance(cpa_anomalies, list) and cpa_anomalies:
                detector.visualize_anomalies('cpa', cpa_anomalies, 
                                           output_file="cpa_anomalies.png", 
                                           campaign=campaign)
                visual_results["cpa_visualization"] = "Generated cpa_anomalies.png"
        
        # If no specific metric was mentioned, run all detections
        if not results:
            spend_anomalies = detector.detect_spend_spikes(column='spend', campaign=campaign)
            results["spend_anomalies"] = spend_anomalies
            
            impression_anomalies = detector.detect_metric_drops(column='impressions', campaign=campaign)
            results["impression_anomalies"] = impression_anomalies
            
            click_anomalies = detector.detect_metric_drops(column='clicks', campaign=campaign)
            results["click_anomalies"] = click_anomalies
            
            ctr_anomalies = detector.detect_ratio_anomalies(
                numerator='clicks', denominator='impressions', campaign=campaign)
            results["ctr_anomalies"] = ctr_anomalies
            
            conv_anomalies = detector.detect_metric_drops(column='conversions', campaign=campaign)
            results["conversion_anomalies"] = conv_anomalies
            
            cpa_anomalies = detector.detect_spend_spikes(column='cpa', campaign=campaign)
            results["cpa_anomalies"] = cpa_anomalies
            
            # Get summary statistics
            summary = detector.get_summary_statistics(campaign=campaign)
            results["summary"] = summary
            
            # Generate visualizations if requested
            if generate_visuals:
                for metric, anomalies in results.items():
                    if isinstance(anomalies, list) and anomalies:
                        col_name = metric.split('_')[0]  # Extract metric name from key
                        if col_name in detector.data.columns:
                            detector.visualize_anomalies(
                                col_name, anomalies,
                                output_file=f"{col_name}_anomalies.png", 
                                campaign=campaign)
                            visual_results[f"{col_name}_visualization"] = f"Generated {col_name}_anomalies.png"
        
        # Add visualization results if any
        if visual_results:
            results["visualizations"] = visual_results
            
        return json.dumps(results, indent=2, cls=NumpyEncoder)
    
    except Exception as e:
        import traceback
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc()
        }, indent=2, cls=NumpyEncoder)


# Define the tools
anomaly_tool = Tool(
    name="AnomalyDetectionTool",
    func=run_anomaly_detection,
    description="Detects anomalies in marketing metrics such as spend spikes, impression drops, click drops, conversion rate changes, etc."
)

def list_available_metrics(query):
    """Function to list available metrics and campaigns in the data"""
    try:
        detector = AnomalyDetector()
        detector.load_data("marketing_metrics.csv")
        
        # Get metrics from column names
        metrics = [col for col in detector.data.columns if col not in ['date', 'campaign']]
        
        # Get unique campaigns
        campaigns = detector.data['campaign'].unique().tolist() if 'campaign' in detector.data.columns else []
        
        # Get date range
        date_range = {
            "start": detector.data['date'].min().strftime('%Y-%m-%d'),
            "end": detector.data['date'].max().strftime('%Y-%m-%d'),
            "days": (detector.data['date'].max() - detector.data['date'].min()).days + 1
        }
        
        result = {
            "available_metrics": metrics,
            "available_campaigns": campaigns,
            "date_range": date_range
        }
        
        return json.dumps(result, indent=2, cls=NumpyEncoder)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2, cls=NumpyEncoder)

metrics_tool = Tool(
    name="ListAvailableMetricsTool",
    func=list_available_metrics,
    description="Lists available metrics and campaigns in the marketing data"
)

# Initialize Azure OpenAI client (fallback to environment variables if available)
try:
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        azure_deployment="gpt-4o",
        api_version="2024-06-01",
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        max_tokens=None,
        temperature=0.7,
    )
    # Initialize agent with Azure OpenAI
    agent = initialize_agent(
        [anomaly_tool, metrics_tool], 
        llm, 
        agent="zero-shot-react-description", 
        verbose=True
    )
except Exception as e:
    print(f"Warning: Could not initialize Azure OpenAI client. ({str(e)})")
    print("Agent functionality will be limited to direct function calls.")
    agent = None

def run_direct_detection(query=None, generate_visuals=False):
    """Run anomaly detection directly without an agent"""
    if not query:
        query = "Check for all anomalies"
        if generate_visuals:
            query += " and visualize them"
    
    result = run_anomaly_detection(query)
    print(result)
    return result

# Example usage
if __name__ == "__main__":
    # Check if we can run with agent, otherwise run direct detection
    try:
        if agent:
            # Run the agent with a query
            response = agent.run("Check for any spend spikes or impression drops in our marketing data")
            print(response)
        else:
            # Run direct detection with visualization
            run_direct_detection(generate_visuals=True)
    except Exception as e:
        print(f"Error running agent: {str(e)}")
        print("Falling back to direct detection...")
        run_direct_detection(generate_visuals=True)

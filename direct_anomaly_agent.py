import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import json
import re
import argparse

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


class RuleBasedAnomalyAgent:
    """A rule-based agent for anomaly detection without LLM integration"""
    
    def __init__(self, data_file="marketing_metrics.csv"):
        """Initialize the agent with a data file"""
        self.detector = AnomalyDetector(lookback_days=7, threshold_multiplier=1.8)
        self.data_file = data_file
        self.load_data()
        
        # Define patterns for command recognition
        self.patterns = {
            "spend_spike": re.compile(r'(?:spend|cost).*(?:spike|increase|high|excessive)', re.IGNORECASE),
            "impression_drop": re.compile(r'impression.*(?:drop|decrease|low|down)', re.IGNORECASE),
            "click_anomaly": re.compile(r'click.*(?:anomaly|drop|spike|unusual)', re.IGNORECASE),
            "conversion_rate": re.compile(r'(?:conversion|cvr).*(?:rate|ratio)', re.IGNORECASE),
            "cpa_spike": re.compile(r'(?:cpa|cost per acquisition).*(?:spike|increase|high)', re.IGNORECASE),
            "campaign_a": re.compile(r'campaign[_\s]*a', re.IGNORECASE),
            "campaign_b": re.compile(r'campaign[_\s]*b', re.IGNORECASE),
            "visualize": re.compile(r'(?:visualize|chart|graph|plot)', re.IGNORECASE),
            "all_anomalies": re.compile(r'(?:all|every).*(?:anomaly|anomalies)', re.IGNORECASE),
            "summary": re.compile(r'summary|statistics|stats|overview', re.IGNORECASE)
        }
        
    def load_data(self):
        """Load the marketing metrics data"""
        data_loaded = self.detector.load_data(self.data_file)
        if not data_loaded:
            print(f"Failed to load marketing metrics data from {self.data_file}")
        
    def get_available_metrics(self):
        """Get information about available metrics and campaigns"""
        if self.detector.data is None:
            return {"error": "No data loaded"}
            
        # Get metrics from column names
        metrics = [col for col in self.detector.data.columns if col not in ['date', 'campaign']]
        
        # Get unique campaigns
        campaigns = (self.detector.data['campaign'].unique().tolist() 
                    if 'campaign' in self.detector.data.columns else [])
        
        # Get date range
        date_range = {
            "start": self.detector.data['date'].min().strftime('%Y-%m-%d'),
            "end": self.detector.data['date'].max().strftime('%Y-%m-%d'),
            "days": (self.detector.data['date'].max() - self.detector.data['date'].min()).days + 1
        }
        
        result = {
            "available_metrics": metrics,
            "available_campaigns": campaigns,
            "date_range": date_range
        }
        
        return result
    
    def format_anomaly_message(self, metric, anomalies):
        """Format anomaly results into readable message"""
        if isinstance(anomalies, dict) and "error" in anomalies:
            return f"Error analyzing {metric}: {anomalies['error']}"
            
        if not anomalies:
            return f"No anomalies detected in {metric}"
            
        messages = [f"Found {len(anomalies)} anomalies in {metric}:"]
        for i, anomaly in enumerate(anomalies[:5]):  # Show only top 5
            date = anomaly.get('date', 'Unknown date')
            campaign = anomaly.get('campaign', 'Unknown campaign')
            value = anomaly.get('value', 0)
            baseline = anomaly.get('baseline_mean', 0)
            
            if 'percent_increase' in anomaly:
                change = f"{anomaly['percent_increase']:.1f}% increase"
            elif 'percent_decrease' in anomaly:
                change = f"{anomaly['percent_decrease']:.1f}% decrease"
            else:
                change = "changed"
                
            messages.append(f"  {i+1}. {date} - {campaign}: {value:.2f} ({change} from baseline {baseline:.2f})")
            
        if len(anomalies) > 5:
            messages.append(f"  ... and {len(anomalies) - 5} more anomalies")
            
        return "\n".join(messages)
    
    def process_command(self, command):
        """Process a command from the user"""
        results = {}
        messages = []
        visual_results = {}
        campaign = None
        generate_visuals = False
        
        # Preprocess command for pattern matching
        command = command.lower().strip()
        
        # Extract campaign information
        if self.patterns["campaign_a"].search(command):
            campaign = "Campaign A"
        elif self.patterns["campaign_b"].search(command):
            campaign = "Campaign B"
            
        # Check if visualization requested
        if self.patterns["visualize"].search(command):
            generate_visuals = True
            
        # Get metrics info if needed
        if "metrics" in command or "available" in command or "help" in command:
            metrics_info = self.get_available_metrics()
            messages.append("Available metrics and campaigns:")
            messages.append(f"Metrics: {', '.join(metrics_info['available_metrics'])}")
            messages.append(f"Campaigns: {', '.join(metrics_info['available_campaigns'])}")
            messages.append(f"Date range: {metrics_info['date_range']['start']} to {metrics_info['date_range']['end']} "
                          f"({metrics_info['date_range']['days']} days)")
        
        # Handle spend spikes
        if self.patterns["spend_spike"].search(command):
            spend_anomalies = self.detector.detect_spend_spikes(column='spend', campaign=campaign)
            results["spend_anomalies"] = spend_anomalies
            messages.append(self.format_anomaly_message("spend", spend_anomalies))
            
            if generate_visuals and isinstance(spend_anomalies, list) and spend_anomalies:
                self.detector.visualize_anomalies('spend', spend_anomalies, 
                                                output_file="spend_anomalies.png", 
                                                campaign=campaign)
                visual_results["spend_visualization"] = "Generated spend_anomalies.png"
                messages.append("Created visualization: spend_anomalies.png")
        
        # Handle impression drops
        if self.patterns["impression_drop"].search(command):
            impression_anomalies = self.detector.detect_metric_drops(column='impressions', campaign=campaign)
            results["impression_anomalies"] = impression_anomalies
            messages.append(self.format_anomaly_message("impressions", impression_anomalies))
            
            if generate_visuals and isinstance(impression_anomalies, list) and impression_anomalies:
                self.detector.visualize_anomalies('impressions', impression_anomalies, 
                                                output_file="impression_anomalies.png", 
                                                campaign=campaign)
                visual_results["impressions_visualization"] = "Generated impression_anomalies.png"
                messages.append("Created visualization: impression_anomalies.png")
                
        # Handle click anomalies
        if self.patterns["click_anomaly"].search(command):
            click_anomalies = self.detector.detect_metric_drops(column='clicks', campaign=campaign)
            results["click_anomalies"] = click_anomalies
            messages.append(self.format_anomaly_message("clicks", click_anomalies))
            
            if generate_visuals and isinstance(click_anomalies, list) and click_anomalies:
                self.detector.visualize_anomalies('clicks', click_anomalies, 
                                               output_file="click_anomalies.png", 
                                               campaign=campaign)
                visual_results["clicks_visualization"] = "Generated click_anomalies.png"
                messages.append("Created visualization: click_anomalies.png")
                
        # Handle conversion related queries
        if "conversion" in command:
            conv_anomalies = self.detector.detect_metric_drops(column='conversions', campaign=campaign)
            results["conversion_anomalies"] = conv_anomalies
            messages.append(self.format_anomaly_message("conversions", conv_anomalies))
            
            if generate_visuals and isinstance(conv_anomalies, list) and conv_anomalies:
                self.detector.visualize_anomalies('conversions', conv_anomalies, 
                                                output_file="conversion_anomalies.png", 
                                                campaign=campaign)
                visual_results["conversions_visualization"] = "Generated conversion_anomalies.png"
                messages.append("Created visualization: conversion_anomalies.png")
                
        # Handle CTR anomalies
        if "ctr" in command or "click-through" in command or "click through" in command:
            ctr_anomalies = self.detector.detect_ratio_anomalies(
                numerator='clicks', denominator='impressions', campaign=campaign)
            results["ctr_anomalies"] = ctr_anomalies
            messages.append(self.format_anomaly_message("click-through rate (CTR)", ctr_anomalies))
            
        # Handle CPA anomalies
        if self.patterns["cpa_spike"].search(command):
            cpa_anomalies = self.detector.detect_spend_spikes(column='cpa', campaign=campaign)
            results["cpa_anomalies"] = cpa_anomalies
            messages.append(self.format_anomaly_message("cost per acquisition (CPA)", cpa_anomalies))
            
            if generate_visuals and isinstance(cpa_anomalies, list) and cpa_anomalies:
                self.detector.visualize_anomalies('cpa', cpa_anomalies, 
                                               output_file="cpa_anomalies.png", 
                                               campaign=campaign)
                visual_results["cpa_visualization"] = "Generated cpa_anomalies.png"
                messages.append("Created visualization: cpa_anomalies.png")
        
        # Handle summary request
        if self.patterns["summary"].search(command):
            summary = self.detector.get_summary_statistics(campaign=campaign)
            results["summary"] = summary
            
            if isinstance(summary, dict) and "error" not in summary:
                campaign_str = f" for {campaign}" if campaign else ""
                messages.append(f"Summary statistics{campaign_str}:")
                messages.append(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']} "
                              f"({summary['date_range']['days']} days)")
                
                for metric, stats in summary['metrics'].items():
                    messages.append(f"  {metric}: mean={stats['mean']:.2f}, median={stats['median']:.2f}, "
                                  f"min={stats['min']:.2f}, max={stats['max']:.2f}, total={stats['total']:.2f}")
            else:
                messages.append(f"Error getting summary: {summary.get('error', 'Unknown error')}")
        
        # Handle all anomalies or default behavior if no specific patterns matched
        if self.patterns["all_anomalies"].search(command) or not results:
            spend_anomalies = self.detector.detect_spend_spikes(column='spend', campaign=campaign)
            results["spend_anomalies"] = spend_anomalies
            messages.append(self.format_anomaly_message("spend", spend_anomalies))
            
            impression_anomalies = self.detector.detect_metric_drops(column='impressions', campaign=campaign)
            results["impression_anomalies"] = impression_anomalies
            messages.append(self.format_anomaly_message("impressions", impression_anomalies))
            
            click_anomalies = self.detector.detect_metric_drops(column='clicks', campaign=campaign)
            results["click_anomalies"] = click_anomalies
            messages.append(self.format_anomaly_message("clicks", click_anomalies))
            
            conv_anomalies = self.detector.detect_metric_drops(column='conversions', campaign=campaign)
            results["conversion_anomalies"] = conv_anomalies
            messages.append(self.format_anomaly_message("conversions", conv_anomalies))
            
            ctr_anomalies = self.detector.detect_ratio_anomalies(
                numerator='clicks', denominator='impressions', campaign=campaign)
            results["ctr_anomalies"] = ctr_anomalies
            messages.append(self.format_anomaly_message("CTR", ctr_anomalies))
            
            cpa_anomalies = self.detector.detect_spend_spikes(column='cpa', campaign=campaign)
            results["cpa_anomalies"] = cpa_anomalies
            messages.append(self.format_anomaly_message("CPA", cpa_anomalies))
            
            if generate_visuals:
                # Generate visualizations for metrics with anomalies
                for metric_key, anomalies in results.items():
                    if isinstance(anomalies, list) and anomalies:
                        metric_name = metric_key.split('_')[0]  # Extract metric name
                        if metric_name in self.detector.data.columns:
                            self.detector.visualize_anomalies(
                                metric_name, anomalies,
                                output_file=f"{metric_name}_anomalies.png",
                                campaign=campaign
                            )
                            visual_results[f"{metric_name}_visualization"] = f"Generated {metric_name}_anomalies.png"
                            messages.append(f"Created visualization: {metric_name}_anomalies.png")
        
        # Add visualization results if any
        if visual_results:
            results["visualizations"] = visual_results
        
        if not messages:
            messages.append("I couldn't understand your request. Try asking about 'spend spikes', 'impression drops', or 'all anomalies'.")
        
        # Return both the structured results and formatted messages
        return {
            "results": results,
            "message": "\n\n".join(messages)
        }

    def run_interactive(self):
        """Run the agent in interactive mode"""
        print("Rule-Based Anomaly Detection Agent")
        print("----------------------------------")
        print("Available commands:")
        print("  - Check for spend spikes")
        print("  - Look for impression drops")
        print("  - Analyze clicks and conversions")
        print("  - Show all anomalies")
        print("  - Visualize [metric] anomalies")
        print("  - Show available metrics")
        print("  - exit/quit")
        print("----------------------------------")
        
        while True:
            command = input("\nEnter command (or 'exit' to quit): ")
            if command.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
                
            try:
                result = self.process_command(command)
                print("\n" + result["message"])
                
                # Offer to save full results to JSON
                if "y" in input("\nSave detailed results to JSON? (y/n): ").lower():
                    filename = f"anomaly_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(filename, 'w') as f:
                        json.dump(result["results"], f, indent=2, cls=NumpyEncoder)
                    print(f"Results saved to {filename}")
            except Exception as e:
                print(f"Error processing command: {str(e)}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Rule-Based Anomaly Detection Agent')
    parser.add_argument('--data', default="marketing_metrics.csv", 
                        help='Path to the marketing metrics CSV file')
    parser.add_argument('--command', help='Command to run (instead of interactive mode)')
    parser.add_argument('--output', help='Output JSON file for results (optional)')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    return parser.parse_args()
    
    
if __name__ == "__main__":
    args = parse_args()
    
    # Initialize the agent
    agent = RuleBasedAnomalyAgent(data_file=args.data)
    
    # Run in interactive mode or process single command
    if args.command:
        # Inject visualization flag into command if requested
        if args.visualize and "visualize" not in args.command.lower():
            args.command += " with visualization"
            
        result = agent.process_command(args.command)
        print(result["message"])
        
        # Save results if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result["results"], f, indent=2, cls=NumpyEncoder)
            print(f"Results saved to {args.output}")
    else:
        # Run in interactive mode
        agent.run_interactive()

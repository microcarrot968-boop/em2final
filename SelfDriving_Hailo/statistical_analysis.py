#!/usr/bin/env python3
"""
Statistical Analysis Module
Handles Week 6-8 statistical analysis of detection data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from typing import List, Tuple, Dict, Any


class StatisticalAnalyzer:
    """Handles statistical analysis of object detection data."""
    
    def __init__(self, csv_path: str = 'run_log.csv'):
        """
        Initialize StatisticalAnalyzer.
        
        Args:
            csv_path (str): Path to CSV log file.
        """
        self.csv_path = csv_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load and preprocess the detection data."""
        try:
            self.df = pd.read_csv(self.csv_path)
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            self.df = self.df.sort_values('timestamp').reset_index(drop=True)
            print(f"Data loaded successfully: {len(self.df)} records")
            print(f"Columns: {list(self.df.columns)}")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            self.df = None
    
    def discrete_distribution_analysis(self, window_size: int = 10, time_window: int = 10):
        """
        Week 6: Analyze discrete distributions (Binomial, Geometric, Poisson).
        
        Args:
            window_size (int): Size of window for binomial analysis.
            time_window (int): Time window in seconds for Poisson analysis.
        """
        if self.df is None:
            print("No data available for analysis.")
            return
        
        print("=== Week 6: Discrete Distribution Analysis ===")
        
        # Binomial Analysis: Number of correct predictions per window
        p_hat = np.mean(self.df['is_correct'])
        bins = [self.df['is_correct'][i:i+window_size].sum() 
                for i in range(0, len(self.df), window_size) 
                if len(self.df)-i >= window_size]
        
        if bins:
            x = np.arange(0, window_size + 1)
            pmf = stats.binom.pmf(x, window_size, p_hat)
            
            plt.figure(figsize=(10, 6))
            plt.hist(bins, bins=range(0, window_size + 2), density=True, 
                    alpha=0.6, label='Empirical', color='skyblue')
            plt.plot(x, pmf, marker='o', label=f'Binomial(n={window_size}, p={p_hat:.3f})', 
                    color='red', linewidth=2)
            plt.xlabel(f'Number of correct predictions in window of {window_size}')
            plt.ylabel('Probability')
            plt.title('Binomial Distribution Analysis')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
            print(f"Binomial Analysis - p_hat: {p_hat:.3f}")
            print(f"Expected successes per window: {window_size * p_hat:.2f}")
        
        # Geometric Analysis: Trials until first success
        def trials_until_success(series):
            """Calculate trials until first success in a series."""
            count = 0
            for value in series:
                count += 1
                if value == 1:
                    return count
            return np.nan
        
        geom_samples = []
        for i in range(0, len(self.df), window_size):
            chunk = self.df['is_correct'][i:i+window_size].tolist()
            if len(chunk) == window_size:
                trials = trials_until_success(chunk)
                if not np.isnan(trials):
                    geom_samples.append(trials)
        
        if geom_samples:
            p_geo = 1.0 / np.mean(geom_samples)
            gx = np.arange(1, window_size + 1)
            gpmf = stats.geom.pmf(gx, p_geo)
            
            plt.figure(figsize=(10, 6))
            plt.hist(geom_samples, bins=range(1, window_size + 2), density=True,
                   alpha=0.6, label='Empirical', color='lightgreen')
            plt.plot(gx, gpmf, marker='o', label=f'Geometric(p={p_geo:.3f})',
                    color='red', linewidth=2)
            plt.xlabel('Trials until first success')
            plt.ylabel('Probability')
            plt.title('Geometric Distribution Analysis')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
            print(f"Geometric Analysis - p: {p_geo:.3f}")
            print(f"Expected trials until success: {1/p_geo:.2f}")
        
        # Poisson Analysis: Detections per time window
        self.df['sec'] = (self.df['timestamp'] - self.df['timestamp'].iloc[0]).dt.total_seconds()
        self.df['bucket'] = (self.df['sec'] // time_window).astype(int)
        counts = self.df.groupby('bucket')['is_correct'].sum()
        
        if len(counts) > 0:
            lam = counts.mean()
            px = np.arange(0, max(5, int(counts.max()) + 2))
            ppmf = stats.poisson.pmf(px, lam)
            
            plt.figure(figsize=(10, 6))
            plt.hist(counts, bins=range(int(counts.max()) + 2), density=True,
                    alpha=0.6, label='Empirical', color='lightcoral')
            plt.plot(px, ppmf, marker='o', label=f'Poisson(λ={lam:.3f})',
                    color='red', linewidth=2)
            plt.xlabel(f'Number of successes per {time_window}s window')
            plt.ylabel('Probability')
            plt.title('Poisson Distribution Analysis')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
            print(f"Poisson Analysis - λ: {lam:.3f}")
            print(f"Expected successes per {time_window}s: {lam:.2f}")
    
    def continuous_distribution_analysis(self):
        """
        Week 7: Analyze continuous distributions with AIC/BIC model selection.
        """
        if self.df is None:
            print("No data available for analysis.")
            return
        
        print("=== Week 7: Continuous Distribution Analysis ===")
        
        def aic(log_likelihood: float, k: int) -> float:
            """Calculate Akaike Information Criterion."""
            return 2 * k - 2 * log_likelihood
        
        def bic(log_likelihood: float, k: int, n: int) -> float:
            """Calculate Bayesian Information Criterion."""
            return k * np.log(n) - 2 * log_likelihood
        
        def fit_and_compare(data: np.ndarray, candidates: List[Tuple[str, Any]]) -> List[Tuple]:
            """Fit multiple distributions and compare using AIC/BIC."""
            data = np.asarray(data)
            data = data[np.isfinite(data)]
            n = len(data)
            
            if n == 0:
                return []
            
            results = []
            for name, distribution in candidates:
                try:
                    params = distribution.fit(data)
                    log_pdf = distribution.logpdf(data, *params)
                    log_likelihood = float(np.sum(log_pdf))
                    k = len(params)
                    results.append((name, params, aic(log_likelihood, k), 
                                  bic(log_likelihood, k, n), log_likelihood))
                except Exception as e:
                    print(f"Error fitting {name}: {str(e)}")
                    continue
            
            return sorted(results, key=lambda x: x[2])  # Sort by AIC
        
        # Confidence analysis
        conf_data = self.df['confidence'].clip(1e-6, 1-1e-6)
        conf_candidates = [('Beta', stats.beta), ('Normal', stats.norm)]
        conf_results = fit_and_compare(conf_data, conf_candidates)
        
        print("Confidence Distribution Analysis:")
        for name, params, aic_val, bic_val, ll in conf_results:
            print(f"  {name}: AIC={aic_val:.2f}, BIC={bic_val:.2f}, params={params}")
        
        # Inference time analysis
        time_data = self.df['inference_time_ms'].clip(1e-6)
        time_candidates = [('Exponential', stats.expon), ('Weibull', stats.weibull_min)]
        time_results = fit_and_compare(time_data, time_candidates)
        
        print("\nInference Time Distribution Analysis:")
        for name, params, aic_val, bic_val, ll in time_results:
            print(f"  {name}: AIC={aic_val:.2f}, BIC={bic_val:.2f}, params={params}")
        
        # Distance analysis (if available)
        if 'distance_cm' in self.df.columns:
            dist_data = self.df['distance_cm'].dropna()
            if len(dist_data) > 0:
                dist_candidates = [('Gaussian', stats.norm), ('Lognormal', stats.lognorm)]
                dist_results = fit_and_compare(dist_data, dist_candidates)
                
                print("\nDistance Distribution Analysis:")
                for name, params, aic_val, bic_val, ll in dist_results:
                    print(f"  {name}: AIC={aic_val:.2f}, BIC={bic_val:.2f}, params={params}")
        
        # Plot best fits
        self._plot_best_fits(conf_data, conf_results, "Confidence")
        self._plot_best_fits(time_data, time_results, "Inference Time")
    
    def _plot_best_fits(self, data: np.ndarray, results: List[Tuple], title: str):
        """Plot the best fitting distribution."""
        if not results:
            return
        
        best_name, best_params, _, _, _ = results[0]
        distribution_map = {
            'Beta': stats.beta,
            'Normal': stats.norm,
            'Exponential': stats.expon,
            'Weibull': stats.weibull_min,
            'Gaussian': stats.norm,
            'Lognormal': stats.lognorm
        }
        
        dist = distribution_map.get(best_name)
        if dist is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # PDF plot
        ax1.hist(data, bins=30, density=True, alpha=0.6, label='Empirical', color='skyblue')
        x_range = np.linspace(np.min(data), np.max(data), 200)
        ax1.plot(x_range, dist.pdf(x_range, *best_params), 
                label=f'{best_name} PDF', color='red', linewidth=2)
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.set_title(f'{title} - Best Fit: {best_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # CDF plot
        sorted_data = np.sort(data)
        empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax2.plot(sorted_data, empirical_cdf, label='Empirical CDF', color='skyblue', linewidth=2)
        ax2.plot(x_range, dist.cdf(x_range, *best_params), 
                label=f'{best_name} CDF', color='red', linewidth=2)
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title(f'{title} CDF - Best Fit: {best_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def joint_distribution_analysis(self):
        """
        Week 8: Analyze joint distributions, correlations, and conditional probabilities.
        """
        if self.df is None:
            print("No data available for analysis.")
            return
        
        print("=== Week 8: Joint Distribution Analysis ===")
        
        # Correlation analysis
        correlation_pairs = []
        if 'distance_cm' in self.df.columns:
            correlation_pairs.append(('confidence', 'distance_cm'))
        correlation_pairs.append(('inference_time_ms', 'is_correct'))
        
        print("Correlation Analysis:")
        for var1, var2 in correlation_pairs:
            if var1 in self.df.columns and var2 in self.df.columns:
                subset = self.df[[var1, var2]].dropna()
                if len(subset) > 2:
                    corr = np.corrcoef(subset[var1], subset[var2])[0, 1]
                    print(f"  Correlation({var1}, {var2}) = {corr:.3f}")
        
        # Joint plot for confidence vs inference time
        plt.figure(figsize=(12, 8))
        sns.jointplot(data=self.df, x='confidence', y='inference_time_ms', 
                     kind='scatter', marginal_kws={'bins': 30})
        plt.suptitle('Confidence vs Inference Time (with marginals)')
        plt.tight_layout()
        plt.show()
        
        # Conditional probabilities
        print("\nConditional Probability Analysis:")
        
        # P(Success and Confidence > 0.8)
        p_success_high_conf = np.mean((self.df['is_correct'] == 1) & (self.df['confidence'] > 0.8))
        print(f"P(Success and Confidence > 0.8) = {p_success_high_conf:.3f}")
        
        # P(Confidence > 0.9 | Lighting = Low)
        if 'light_level' in self.df.columns:
            low_light = self.df[self.df['light_level'].astype(str).str.lower() == 'low']
            if len(low_light) > 0:
                p_high_conf_low_light = np.mean(low_light['confidence'] > 0.9)
                print(f"P(Confidence > 0.9 | Lighting = Low) = {p_high_conf_low_light:.3f}")
        
        # P(Success | High Confidence)
        high_conf = self.df[self.df['confidence'] > 0.8]
        if len(high_conf) > 0:
            p_success_given_high_conf = np.mean(high_conf['is_correct'] == 1)
            print(f"P(Success | Confidence > 0.8) = {p_success_given_high_conf:.3f}")
    
    def export_cleaned_data(self, output_path: str = 'run_log_clean.csv'):
        """
        Export cleaned data with quality control flags.
        
        Args:
            output_path (str): Path for cleaned data output.
        """
        if self.df is None:
            print("No data available for export.")
            return
        
        print("=== Exporting Cleaned Data ===")
        
        df_clean = self.df.copy()
        df_clean = df_clean.sort_values('timestamp')
        
        # Calculate estimated FPS and quality control flags
        fps_est = 1000.0 / df_clean['inference_time_ms'].replace(0, np.nan)
        median_fps = fps_est.median()
        std_fps = fps_est.std()
        
        # Flag outliers (high FPS with incorrect predictions)
        df_clean['qc_flag'] = ((fps_est > median_fps + 2 * std_fps) & 
                              (df_clean['is_correct'] == 0)).astype(int)
        
        # Save cleaned data
        df_clean.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        print(f"Quality control flags: {df_clean['qc_flag'].sum()} outliers flagged")
        
        return df_clean


if __name__ == "__main__":
    print("=== Statistical Analysis Test ===")
    
    analyzer = StatisticalAnalyzer()
    
    if analyzer.df is not None:
        print("Running all analyses...")
        analyzer.discrete_distribution_analysis()
        analyzer.continuous_distribution_analysis()
        analyzer.joint_distribution_analysis()
        analyzer.export_cleaned_data()
    else:
        print("No data file found. Please run object detection first.")

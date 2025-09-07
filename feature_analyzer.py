import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import f_oneway
import yaml

class FeatureAnalyzer:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create analysis directory
        self.analysis_dir = os.path.join(self.config['paths']['output_dir'], "analysis")
        os.makedirs(self.analysis_dir, exist_ok=True)
        
        # Set random seed
        np.random.seed(self.config['project']['seed'])
        
        # Fault type mapping
        self.fault_labels = {
            "B007": 'Ball Fault (0.007")',
            "B014": 'Ball Fault (0.014")',
            "B021": 'Ball Fault (0.021")',
            "IR007": 'Inner Race Fault (0.007")',
            "IR014": 'Inner Race Fault (0.014")',
            "IR021": 'Inner Race Fault (0.021")',
            "OR007": 'Outer Race Fault (0.007")',
            "OR014": 'Outer Race Fault (0.014")',
            "OR021": 'Outer Race Fault (0.021")',
            "Normal": 'Healthy'
        }
        
        # Color palette
        self.palette = [
            "#E53935", "#00897B", "#FF6D00", "#FFB300", "#E53935",
            "#8E24AA", "#AB47BC", "#D81B60", "#0B3D91", "#00BFA5"
        ]
    
    def map_fault_label(self, fault_name):
        """Map fault name to friendly English label"""
        if fault_name is None:
            return fault_name
        if "Normal" in fault_name or "normal" in fault_name.lower():
            return self.fault_labels["Normal"]
        for key in self.fault_labels:
            if fault_name.startswith(key):
                return self.fault_labels[key]
        # Fallback by prefix
        if fault_name.startswith("B"):
            return 'Ball Fault'
        if fault_name.startswith("IR"):
            return 'Inner Race Fault'
        if fault_name.startswith("OR"):
            return 'Outer Race Fault'
        return fault_name
    
    def safe_savefig(self, fig, filepath, dpi=300):
        """Safely save figure with error handling"""
        try:
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Error saving {filepath}: {e}")
            # Try alternative name
            alt_path = filepath.replace('.png', '_backup.png')
            try:
                fig.savefig(alt_path, dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved to alternative location: {alt_path}")
            except Exception as e2:
                print(f"Also failed to save to alternative: {e2}")
                plt.close(fig)
    
    def analyze_features(self):
        """Perform comprehensive feature analysis"""
        # Load data
        data_path = os.path.join(self.config['paths']['processed_dir'], "dataset.csv")
        df = pd.read_csv(data_path)
        X = df.drop(columns=['fault'])
        y = df['fault']
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
        
        # Train Random Forest for feature importance
        model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Feature importance
        importances = model.feature_importances_
        features = X.columns
        indices = np.argsort(importances)[::-1]
        
        ranking_df = pd.DataFrame({
            'feature': features[indices],
            'importance': importances[indices],
            'rank': np.arange(1, len(features) + 1)
        })
        ranking_df.to_csv(os.path.join(self.analysis_dir, 'feature_importance_ranking.csv'), index=False)
        
        # Feature importance plot
        plt.figure(figsize=(16, 8))
        plt.bar(range(len(features)), importances[indices], color='teal', edgecolor='black')
        plt.xticks(range(len(features)), features[indices], rotation=90)
        plt.title("Feature Importance (Time + Frequency Domain)")
        plt.ylabel("Relative Importance")
        plt.tight_layout()
        self.safe_savefig(plt, os.path.join(self.analysis_dir, 'feature_importance.png'))
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f'Confusion Matrix (Accuracy: {acc:.2%})')
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        self.safe_savefig(plt, os.path.join(self.analysis_dir, 'confusion_matrix.png'))
        
        # Save classification report
        report_path = os.path.join(self.analysis_dir, "random_forest_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Random Forest - Time + Frequency Features\n")
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write(classification_report(y_test, y_pred, target_names=le.classes_))
            f.write("\nConfusion Matrix:\n")
            f.write(np.array2string(conf_matrix))
        
        # ANOVA analysis
        anova_results = []
        print("Performing ANOVA analysis...")
        
        for col in X.columns:
            groups = [df[df['fault'] == fault_class][col] for fault_class in df['fault'].unique()]
            try:
                fval, pval = f_oneway(*groups)
            except Exception:
                fval, pval = np.nan, np.nan
            anova_results.append({'feature': col, 'f_value': fval, 'p_value': pval})
        
        anova_df = pd.DataFrame(anova_results)
        anova_df.to_csv(os.path.join(self.analysis_dir, 'anova_results.csv'), index=False)
        
        # Filter features by p-value
        p_threshold = 0.05
        significant_anova_df = anova_df[anova_df['p_value'] < p_threshold]
        
        # Create color mapping for fault types
        classes = list(df['fault'].unique())
        color_map = {fc: self.palette[i % len(self.palette)] for i, fc in enumerate(classes)}
        
        # Plot best and worst features by ANOVA
        if not significant_anova_df.empty:
            significant_anova_df_sorted = significant_anova_df.sort_values('p_value', ascending=True)
            top_feature = significant_anova_df_sorted.iloc[0]['feature']
            bottom_feature = significant_anova_df_sorted.iloc[-1]['feature']
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            for fault_class in df['fault'].unique():
                subset = df[df['fault'] == fault_class]
                label = self.map_fault_label(fault_class)
                sns.histplot(subset[top_feature], kde=True, label=label, alpha=0.5, ax=ax1,
                             color=color_map.get(fault_class))
            ax1.set_xlabel(top_feature)
            ax1.set_ylabel('Frequency')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            for fault_class in df['fault'].unique():
                subset = df[df['fault'] == fault_class]
                label = self.map_fault_label(fault_class)
                sns.histplot(subset[bottom_feature], kde=True, label=label, alpha=0.5, ax=ax2,
                             color=color_map.get(fault_class))
            ax2.set_xlabel(bottom_feature)
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            self.safe_savefig(plt, os.path.join(self.analysis_dir, 'anova_features_comparison.png'), dpi=600)
        
        # Pairplot with top 6 RF features
        top_features = features[indices[:6]]
        df_top = df[top_features.tolist() + ['fault']]
        pair = sns.pairplot(df_top, hue='fault', diag_kind='kde', plot_kws={'alpha': 0.6})
        plt.suptitle("Pairplot - Top 6 Features", y=1.02)
        plt.tight_layout()
        self.safe_savefig(pair.fig, os.path.join(self.analysis_dir, 'pairplot_top6_features.png'))
        
        print(f"Analysis completed. Results in: {self.analysis_dir}")

if __name__ == "__main__":
    analyzer = FeatureAnalyzer()
    analyzer.analyze_features()
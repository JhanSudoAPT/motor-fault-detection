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
        
        self.analysis_dir = os.path.join(self.config['paths']['output_dir'], "analysis")
        os.makedirs(self.analysis_dir, exist_ok=True)
        
        np.random.seed(self.config['project']['seed'])
        self.palette = sns.color_palette("tab10")
    
    def safe_savefig(self, fig, filepath, dpi=300):
        try:
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Error saving {filepath}: {e}")
            plt.close(fig)
    
    def analyze_features(self):
        # Load dataset
        data_path = os.path.join(self.config['paths']['processed_dir'], "dataset.csv")
        df = pd.read_csv(data_path)
        X = df.drop(columns=['fault'])
        y = df['fault']
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
        
        # Random Forest
        model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Feature importance
        importances = model.feature_importances_
        features = X.columns
        indices = np.argsort(importances)[::-1]
        ranking_df = pd.DataFrame({'feature': features[indices], 'importance': importances[indices]})
        ranking_df.to_csv(os.path.join(self.analysis_dir, 'feature_importance_ranking.csv'), index=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(features)), importances[indices], color='teal', edgecolor='black')
        plt.xticks(range(len(features)), features[indices], rotation=90)
        plt.title("Feature Importance")
        plt.ylabel("Importance")
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
        
        # Classification report
        report_path = os.path.join(self.analysis_dir, "classification_report.txt")
        with open(report_path, 'w') as f:
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write(classification_report(y_test, y_pred, target_names=le.classes_))
        
        # ANOVA
        anova_results = []
        for col in X.columns:
            groups = [df[df['fault'] == cls][col] for cls in df['fault'].unique()]
            try:
                fval, pval = f_oneway(*groups)
            except:
                fval, pval = np.nan, np.nan
            anova_results.append({'feature': col, 'f_value': fval, 'p_value': pval})
        
        anova_df = pd.DataFrame(anova_results)
        anova_df.to_csv(os.path.join(self.analysis_dir, 'anova_results.csv'), index=False)
        
        # Pairplot top features
        top_features = features[indices[:6]]
        df_top = df[top_features.tolist() + ['fault']]
        pair = sns.pairplot(df_top, hue='fault', diag_kind='kde', plot_kws={'alpha': 0.6})
        plt.suptitle("Pairplot - Top Features", y=1.02)
        plt.tight_layout()
        self.safe_savefig(pair.fig, os.path.join(self.analysis_dir, 'pairplot_top_features.png'))
        
        print(f"Analysis completed. Results saved in {self.analysis_dir}")

if __name__ == "__main__":
    analyzer = FeatureAnalyzer()
    analyzer.analyze_features()

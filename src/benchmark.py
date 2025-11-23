import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from features import extract_features
from Bio import SeqIO

def evaluate(ground_truth_file, prediction_file, output_prefix):
    """
    Evaluate predictions against ground truth.
    """
    print("Loading data...")
    preds = pd.read_csv(prediction_file, sep='\t')
    
    # Infer ground truth from GeneID
    # IDs are like host_123, foreign_456, ameliorated_789
    def get_label(gene_id):
        if "host" in gene_id:
            return 0
        elif "ameliorated" in gene_id:
            return 1
        elif "foreign" in gene_id:
            return 2
        return -1
        
    y_true = preds['GeneID'].apply(get_label)
    y_pred = preds['State']
    
    # Filter out unknown labels
    mask = y_true != -1
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    preds = preds[mask]
    
    print("\nClassification Report (0=Host, 1=Ameliorated, 2=Foreign):")
    print(classification_report(y_true, y_pred, target_names=['Host', 'Ameliorated', 'Foreign']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Host', 'Ameliorated', 'Foreign'],
                yticklabels=['Host', 'Ameliorated', 'Foreign'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f"{output_prefix}_confusion_matrix.png")
    plt.close()
    
    # ROC Curve (Host vs Non-Host)
    y_true_binary = (y_true != 0).astype(int)
    y_prob_non_host = preds['Prob_Ameliorated'] + preds['Prob_Foreign']
    
    auc = roc_auc_score(y_true_binary, y_prob_non_host)
    print(f"\nROC AUC (Host vs Non-Host): {auc:.4f}")
    
    fpr, tpr, _ = roc_curve(y_true_binary, y_prob_non_host)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Host vs Non-Host)')
    plt.legend()
    plt.savefig(f"{output_prefix}_roc_curve.png")
    plt.close()

def visualize_pca(fasta_file, prediction_file, output_prefix):
    """
    Generate PCA plot of features colored by predicted state.
    """
    print(f"Loading sequences from {fasta_file} for PCA...")
    records = list(SeqIO.parse(fasta_file, "fasta"))
    sequences = [str(r.seq) for r in records]
    ids = [r.id for r in records]
    
    print("Extracting features for PCA...")
    # Note: We only use composition features for PCA to keep it simple
    features = np.array([extract_features(s)[0] for s in sequences])
    
    # Load predictions to color points
    preds = pd.read_csv(prediction_file, sep='\t')
    # Map IDs to States
    id_to_state = dict(zip(preds['GeneID'], preds['State']))
    states = [id_to_state.get(gid, 0) for gid in ids]
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(features)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=states, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Predicted State (0=Host, 1=Ameliorated, 2=Foreign)')
    plt.title('PCA of Gene Features')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(f"{output_prefix}_pca.png")
    plt.close()
    print(f"PCA plot saved to {output_prefix}_pca.png")

def visualize_posterior_heatmap(prediction_file, output_prefix):
    """
    Generate a heatmap of posterior probabilities across the genome.
    """
    preds = pd.read_csv(prediction_file, sep='\t')
    
    # Extract probabilities
    probs = preds[['Prob_Host', 'Prob_Ameliorated', 'Prob_Foreign']].values.T
    
    plt.figure(figsize=(15, 4))
    sns.heatmap(probs, cmap='YlOrRd', yticklabels=['Host', 'Ameliorated', 'Foreign'], cbar_kws={'label': 'Probability'})
    plt.xlabel('Gene Index')
    plt.title('Posterior Probabilities along Genome')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_heatmap.png")
    plt.close()
    print(f"Heatmap saved to {output_prefix}_heatmap.png")

def visualize_gene_probabilities(prediction_file, output_prefix, top_n=20):
    """
    Generate a bar plot for the top candidate foreign genes.
    """
    preds = pd.read_csv(prediction_file, sep='\t')
    
    # Sort by Foreign probability
    top_foreign = preds.sort_values('Prob_Foreign', ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 6))
    
    # Stacked bar chart
    x = range(len(top_foreign))
    p1 = plt.bar(x, top_foreign['Prob_Host'], label='Host', color='blue', alpha=0.6)
    p2 = plt.bar(x, top_foreign['Prob_Ameliorated'], bottom=top_foreign['Prob_Host'], label='Ameliorated', color='green', alpha=0.6)
    p3 = plt.bar(x, top_foreign['Prob_Foreign'], bottom=top_foreign['Prob_Host']+top_foreign['Prob_Ameliorated'], label='Foreign', color='red', alpha=0.6)
    
    plt.xticks(x, top_foreign['GeneID'], rotation=45, ha='right')
    plt.ylabel('Probability')
    plt.title(f'Top {top_n} Predicted Foreign Genes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_barplot.png")
    plt.close()
    print(f"Bar plot saved to {output_prefix}_barplot.png")

def main():
    parser = argparse.ArgumentParser(description="Benchmark HGT Detection")
    parser.add_argument('--predictions', required=True, help='Prediction TSV file')
    parser.add_argument('--output', required=True, help='Output prefix for plots')
    parser.add_argument('--fasta', help='Input FASTA file (optional, required for PCA plot)')
    
    args = parser.parse_args()
    
    evaluate(None, args.predictions, args.output)
    
    if args.fasta:
        visualize_pca(args.fasta, args.predictions, args.output)
        
    visualize_posterior_heatmap(args.predictions, args.output)
    visualize_gene_probabilities(args.predictions, args.output)

if __name__ == "__main__":
    main()

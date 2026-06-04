

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns


class ExplainabilityModule:
   
    
    def __init__(self, cloud_feature_names=None):
        
        self.cloud_feature_names = cloud_feature_names or [
            'cloud_mean', 'cloud_std', 'cloud_max', 'cloud_min',
            'cloud_cover', 'brightness', 'texture', 'cloud_density'
        ]
    
    def analyze_feature_importance(self, attention_weights, cloud_features):
        
        batch_size, n_heads, seq_len, _ = attention_weights.shape
        
       
        avg_attention = attention_weights.mean(dim=1).mean(dim=1)  
        
       
        feature_importance = []
        for j in range(cloud_features.shape[-1]):
           
            feature_values = cloud_features[:, :, j]  
            weighted_importance = (avg_attention * feature_values).sum() / (feature_values.sum() + 1e-8)
            feature_importance.append(weighted_importance.item())
        
       
        feature_importance = np.array(feature_importance)
        feature_importance = feature_importance / (feature_importance.sum() + 1e-8)
        
        
        feature_importance_dict = {
            self.cloud_feature_names[i]: float(importance)
            for i, importance in enumerate(feature_importance)
        }
        
        return feature_importance, feature_importance_dict
    
    def analyze_temporal_attention(self, attention_weights):
      
       
        avg_attention = attention_weights.mean(dim=1)  
        
      
        temporal_importance = avg_attention.sum(dim=2).mean(dim=0)  
        
        return temporal_importance.cpu().numpy()
    
    def analyze_semantic_clustering(self, cloud_encodings, n_clusters=3):
       
        batch_size, seq_len, encoding_dim = cloud_encodings.shape
        encodings_flat = cloud_encodings.reshape(-1, encoding_dim).cpu().numpy()
      
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(encodings_flat)
        cluster_centers = kmeans.cluster_centers_
        
       
        cluster_features = {}
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_size = cluster_mask.sum()
            cluster_features[f'cluster_{i}'] = {
                'size': cluster_size,
                'percentage': cluster_size / len(cluster_labels) * 100,
                'center': cluster_centers[i],
            }
        
        return cluster_labels, cluster_centers, cluster_features
    
    def compute_gradients(self, model, cloud_features, target_output_idx=0):
       
        cloud_features = cloud_features.clone().requires_grad_(True)
        
       
        model.eval()
        output = model(cloud_features)
        
        
        if output.dim() > 2:
            target = output[:, target_output_idx, :].sum()  
        else:
            target = output.sum()
        
        gradients = torch.autograd.grad(
            target, cloud_features,
            create_graph=False, retain_graph=False
        )[0]
        
       
        gradient_importance = gradients.abs().mean(dim=(0, 1)).cpu().numpy() 
        
        return gradients.cpu().detach().numpy(), gradient_importance
    
    def counterfactual_analysis(self, model, base_cloud_features, base_prediction,
                               feature_idx, perturbation_range=0.1):
       
        results = {}
        
        model.eval()
        with torch.no_grad():
           
            perturbed_features_pos = base_cloud_features.clone()
            perturbed_features_pos[:, :, feature_idx] *= (1 + perturbation_range)
            pred_pos = model(perturbed_features_pos)
            
            
            perturbed_features_neg = base_cloud_features.clone()
            perturbed_features_neg[:, :, feature_idx] *= (1 - perturbation_range)
            pred_neg = model(perturbed_features_neg)
            
           
            change_pos = (pred_pos - base_prediction) / base_prediction * 100
            change_neg = (pred_neg - base_prediction) / base_prediction * 100
            
            results[f'{self.cloud_feature_names[feature_idx]}_+{perturbation_range*100:.0f}%'] = {
                'prediction': pred_pos.cpu().numpy(),
                'change_percent': change_pos.mean().item()
            }
            
            results[f'{self.cloud_feature_names[feature_idx]}_{-perturbation_range*100:.0f}%'] = {
                'prediction': pred_neg.cpu().numpy(),
                'change_percent': change_neg.mean().item()
            }
        
        return results
    
    def generate_explainability_report(self, model, cloud_features, attention_weights,
                                      prediction, ground_truth=None):
       
        report = {}
        
       
        feature_importance, feature_importance_dict = self.analyze_feature_importance(
            attention_weights, cloud_features
        )
        report['feature_importance'] = {
            'scores': feature_importance_dict,
            'top_features': sorted(
                feature_importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
        }
        
      
        temporal_importance = self.analyze_temporal_attention(attention_weights)
        report['temporal_attention'] = {
            'importance': temporal_importance.tolist(),
            'top_time_steps': np.argsort(temporal_importance)[-3:].tolist()
        }
        
       
        gradients, gradient_importance = self.compute_gradients(
            model, cloud_features
        )
        gradient_dict = {
            self.cloud_feature_names[i]: float(grad)
            for i, grad in enumerate(gradient_importance)
        }
        report['gradient_analysis'] = {
            'importance': gradient_dict,
            'top_features': sorted(
                gradient_dict.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]
        }
        
     
        if ground_truth is not None:
            base_prediction = prediction
            counterfactual_results = {}
            
            top_features = sorted(
                feature_importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            for feature_name, _ in top_features:
                feature_idx = self.cloud_feature_names.index(feature_name)
                results = self.counterfactual_analysis(
                    model, cloud_features, base_prediction,
                    feature_idx, perturbation_range=0.1
                )
                counterfactual_results[feature_name] = results
            
            report['counterfactual_analysis'] = counterfactual_results
        
        return report
    
    def visualize_explainability(self, report, save_path=None):
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


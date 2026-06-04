
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FeatureMappingNetwork(nn.Module):
    
    def __init__(self, cloud_feature_dim=8, weather_feature_dim=7, hidden_dim=64):
        super(FeatureMappingNetwork, self).__init__()
        
        self.mapping = nn.Sequential(
            nn.Linear(cloud_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, weather_feature_dim)
        )
        
    def forward(self, cloud_features):
       
        return self.mapping(cloud_features)


class CloudEncoder(nn.Module):
   
    def __init__(self, cloud_feature_dim=8, embed_dim=128, num_layers=2):
        super(CloudEncoder, self).__init__()
        
        self.input_proj = nn.Linear(cloud_feature_dim, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, cloud_features):
       
        x = self.input_proj(cloud_features)
        x = self.encoder(x)
        x = self.output_proj(x)
        return x


class DistributedPVPredictor(nn.Module):
  
    def __init__(self, configs, strategy='mapping'):
        
        super(DistributedPVPredictor, self).__init__()
        
        self.strategy = strategy
        self.weather_feature_dim = configs.enc_in - 8  
        self.cloud_feature_dim = 8
        self.d_model = configs.d_model
        
        if strategy == 'mapping' or strategy == 'hybrid':
            
            self.feature_mapper = FeatureMappingNetwork(
                cloud_feature_dim=self.cloud_feature_dim,
                weather_feature_dim=self.weather_feature_dim,
                hidden_dim=64
            )
        elif strategy == 'fusion':
            
            self.cloud_encoder = CloudEncoder(
                cloud_feature_dim=self.cloud_feature_dim,
                embed_dim=self.d_model // 2,
                num_layers=2
            )
            self.feature_fusion = nn.Linear(
                self.cloud_feature_dim + self.d_model // 2,
                configs.d_model
            )
        
        
        
        from model.iTransformer import Model as iTransformer
        self.itransformer = iTransformer(configs)
        
    def forward(self, cloud_features, x_mark_enc=None, x_mark_dec=None, mode='predict'):
       
        batch_size, seq_len, _ = cloud_features.shape
        
        if self.strategy == 'mapping' or self.strategy == 'hybrid':
           
            virtual_weather = self.feature_mapper(cloud_features)
            
           
            combined_features = torch.cat([virtual_weather, cloud_features], dim=-1)
           
            
          
           
            dec_inp = torch.zeros(batch_size, self.itransformer.pred_len, combined_features.shape[-1])
            dec_inp = torch.cat([combined_features[:, -self.itransformer.label_len:, :], dec_inp], dim=1)
            
            pred = self.itransformer.forecast(
                combined_features, x_mark_enc, dec_inp, x_mark_dec
            )
            
            if mode == 'train':
                return pred, virtual_weather
            else:
                return pred
                
        elif self.strategy == 'fusion':
           
            cloud_embedding = self.cloud_encoder(cloud_features)
            
           
            fused_features = torch.cat([cloud_features, cloud_embedding], dim=-1)
            fused_features = self.feature_fusion(fused_features)
            
          
            dec_inp = torch.zeros(batch_size, self.itransformer.pred_len, fused_features.shape[-1])
            dec_inp = torch.cat([fused_features[:, -self.itransformer.label_len:, :], dec_inp], dim=1)
            
            pred = self.itransformer.forecast(
                fused_features, x_mark_enc, dec_inp, x_mark_dec
            )
            
            return pred


class DistributedPVTrainer:
    
    def __init__(self, configs, strategy='mapping'):
        self.configs = configs
        self.strategy = strategy
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
       
        self.model = DistributedPVPredictor(configs, strategy=strategy).to(self.device)
        
       
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=configs.learning_rate)
        
      
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
    def train_mapping_network(self, centralized_data_loader):
        
        self.model.train()
        total_loss = 0
        mapping_loss = 0
        pred_loss = 0
        
        for batch_x, batch_y, batch_x_mark, batch_y_mark in centralized_data_loader:
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device) if batch_x_mark is not None else None
            batch_y_mark = batch_y_mark.float().to(self.device) if batch_y_mark is not None else None
            
          
           
            cloud_features = batch_x[:, :, -8:]  # [batch, seq_len, 8]
            weather_features = batch_x[:, :, :-8]  # [batch, seq_len, 7]
            
           
            self.optimizer.zero_grad()
            
           
            virtual_weather = self.model.feature_mapper(cloud_features)
            
         
            mapping_loss_batch = self.mse_loss(virtual_weather, weather_features)
            
           
            combined_features = torch.cat([virtual_weather, cloud_features], dim=-1)
            dec_inp = torch.zeros_like(batch_y[:, -self.configs.pred_len:, :])
            dec_inp = torch.cat([batch_y[:, :self.configs.label_len, :], dec_inp], dim=1)
            
            pred, _ = self.model(cloud_features, batch_x_mark, batch_y_mark, mode='train')
            
          
            pred_loss_batch = self.mse_loss(
                pred[:, -self.configs.pred_len:, :],
                batch_y[:, -self.configs.pred_len:, :]
            )
            
        
            loss = mapping_loss_batch + pred_loss_batch
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            mapping_loss += mapping_loss_batch.item()
            pred_loss += pred_loss_batch.item()
        
        return {
            'total_loss': total_loss / len(centralized_data_loader),
            'mapping_loss': mapping_loss / len(centralized_data_loader),
            'pred_loss': pred_loss / len(centralized_data_loader)
        }
    
    def train_with_distributed_data(self, distributed_data_loader):
        
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y, batch_x_mark, batch_y_mark in distributed_data_loader:
           
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device) if batch_x_mark is not None else None
            batch_y_mark = batch_y_mark.float().to(self.device) if batch_y_mark is not None else None
            
            self.optimizer.zero_grad()
            
         
            pred = self.model(batch_x, batch_x_mark, batch_y_mark, mode='predict')
            
         
            loss = self.mse_loss(
                pred[:, -self.configs.pred_len:, :],
                batch_y[:, -self.configs.pred_len:, :]
            )
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(distributed_data_loader)
    
    def predict(self, cloud_features, x_mark_enc=None, x_mark_dec=None):
    
        self.model.eval()
        with torch.no_grad():
            cloud_features = torch.FloatTensor(cloud_features).to(self.device)
            if x_mark_enc is not None:
                x_mark_enc = torch.FloatTensor(x_mark_enc).to(self.device)
            if x_mark_dec is not None:
                x_mark_dec = torch.FloatTensor(x_mark_dec).to(self.device)
            
            pred = self.model(cloud_features, x_mark_enc, x_mark_dec, mode='predict')
            return pred.cpu().numpy()


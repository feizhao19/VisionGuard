# Modified code for CustomCLIPModel to interact with all transformer layers in CLIP

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class CustomCLIPModel(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32"):
        super(CustomCLIPModel, self).__init__()
        
        # Load the pre-trained CLIP model with hidden states enabled
        self.clip_model = CLIPModel.from_pretrained(clip_model_name, output_hidden_states=True)
        
        # Freeze the CLIP model parameters (optional)
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Define the query tokens for Q-Former (learnable parameters)
        self.num_query_tokens = 1
        self.query_tokens = nn.Parameter(torch.randn(1, self.num_query_tokens, self.clip_model.config.projection_dim))
        
        # Define a cross-attention layer to interact with each transformer layer in the vision and text encoders
        self.cross_attention = nn.MultiheadAttention(self.clip_model.config.projection_dim, num_heads=8)

        # Define the binary classification subnetwork
        self.binary_classifier = nn.Sequential(
            nn.Linear(self.clip_model.config.projection_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )


        # Projection layers for each block, excluding the last one
        self.projection_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.clip_model.vision_model.config.hidden_size),
                nn.Linear(self.clip_model.vision_model.config.hidden_size, self.clip_model.config.projection_dim)
            ) for _ in range(len(self.clip_model.vision_model.encoder.layers) - 1)
        ])
        # print("the layers number:", self.clip_model.vision_model.encoder.layers)


    def extract_text_embeds(self, text_inputs):
        """ Extracts the text embeddings using the CLIP text model. """
        text_output = self.clip_model.text_model(
            input_ids=text_inputs['input_ids'], 
            attention_mask=text_inputs['attention_mask'],
            output_hidden_states=True
        )
        text_hidden_states = text_output.hidden_states
        if text_hidden_states is None:
            raise ValueError("Hidden states for text embeddings are not available.")
        return text_hidden_states

    def extract_image_embeds(self, image_inputs):
        """ Extracts the image embeddings using the CLIP vision model. """
        image_output = self.clip_model.vision_model(
            pixel_values=image_inputs,
            output_hidden_states=True
        )
        image_hidden_states = image_output.hidden_states
        if image_hidden_states is None:
            raise ValueError("Hidden states for image embeddings are not available.")
        return image_hidden_states

    def forward(self, image_inputs, text_inputs_positive, text_inputs_negative):
        # Extract image and text embeddings (full token-level embeddings from all layers)
        image_hidden_states = self.extract_image_embeds(image_inputs)  # List of hidden states from each layer
        text_hidden_states_positive = self.extract_text_embeds(text_inputs_positive)
        text_hidden_states_negative = self.extract_text_embeds(text_inputs_negative)
        
        query_tokens_pos = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_tokens_neg = query_tokens_pos.clone()  # Negative query is a duplicate of the positive query

        # Loop over all transformer layers and apply cross-attention
        # print("the length of the image_hidden_states", len(image_hidden_states))
        for i in range(1, len(image_hidden_states)):
            # Get the hidden states from the i-th transformer layer
            image_features = image_hidden_states[i]
            text_features_pos = text_hidden_states_positive[i]
            text_features_neg = text_hidden_states_negative[i]
            
            # Project image features to match the projection dimension if needed
            if i == (len(image_hidden_states) - 1):
                # print("i idx number for the last", i)
                image_features = self.clip_model.vision_model.post_layernorm(image_features)
                image_features = self.clip_model.visual_projection(image_features)
            else:
                # print("i idx number", i)
                image_features = self.projection_layers[i -1](image_features)

            # Combine image and text features
            combined_features_pos = torch.cat([image_features, text_features_pos], dim=1)
            combined_features_neg = torch.cat([image_features, text_features_neg], dim=1)
            
            # Apply cross-attention for positive captions
            combined_features_pos = combined_features_pos.permute(1, 0, 2)  # Shape: [seq_len, batch_size, projection_dim]
            query_tokens_pos = query_tokens_pos.permute(1, 0, 2)  # Shape: [num_query_tokens, batch_size, projection_dim]
            query_tokens_pos, _ = self.cross_attention(query_tokens_pos, combined_features_pos, combined_features_pos)
            query_tokens_pos = query_tokens_pos.permute(1, 0, 2)  # Shape: [batch_size, num_query_tokens, projection_dim]
            
            # Apply cross-attention for negative captions
            combined_features_neg = combined_features_neg.permute(1, 0, 2)
            query_tokens_neg = query_tokens_neg.permute(1, 0, 2)
            query_tokens_neg, _ = self.cross_attention(query_tokens_neg, combined_features_neg, combined_features_neg)
            query_tokens_neg = query_tokens_neg.permute(1, 0, 2)
        
        # Normalize embeddings
        query_tokens_attended_pos = query_tokens_pos / query_tokens_pos.norm(p=2, dim=-1, keepdim=True)
        query_tokens_attended_neg = query_tokens_neg / query_tokens_neg.norm(p=2, dim=-1, keepdim=True)
        
        # Cosine similarity logits
        logits_query_pos = torch.matmul(query_tokens_attended_pos, image_features[:, 0, :].unsqueeze(-1)).squeeze(-1) 
        logits_query_neg = torch.matmul(query_tokens_attended_neg, image_features[:, 0, :].unsqueeze(-1)).squeeze(-1) 

        # Binary classification logits
        logits_binary_classification_pos = self.binary_classifier(query_tokens_attended_pos.squeeze(0))
        logits_binary_classification_neg = self.binary_classifier(query_tokens_attended_neg.squeeze(0))

        return _, _, logits_binary_classification_pos, logits_binary_classification_neg, logits_query_pos, logits_query_neg
    
    def infer(self, image_inputs, text_inputs):
        # Extract image and text embeddings
        image_hidden_states = self.extract_image_embeds(image_inputs)
        text_hidden_states = self.extract_text_embeds(text_inputs)
        
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)

        # Loop over all transformer layers and apply cross-attention
        for i in range(1, len(image_hidden_states)):
            # Get the hidden states from the i-th transformer layer
            image_features = image_hidden_states[i]
            text_features = text_hidden_states[i]
            
            # Project image features to match the projection dimension if needed
            if i == (len(image_hidden_states) - 1):
                image_features = self.clip_model.vision_model.post_layernorm(image_features)
                image_features = self.clip_model.visual_projection(image_features)
            else:
                image_features = self.projection_layers[i - 1](image_features)

            # Combine image and text features
            combined_features = torch.cat([image_features, text_features], dim=1)
            
            # Apply cross-attention
            combined_features = combined_features.permute(1, 0, 2)  # Shape: [seq_len, batch_size, projection_dim]
            query_tokens = query_tokens.permute(1, 0, 2)  # Shape: [num_query_tokens, batch_size, projection_dim]
            query_tokens, _ = self.cross_attention(query_tokens, combined_features, combined_features)
            query_tokens = query_tokens.permute(1, 0, 2)  # Shape: [batch_size, num_query_tokens, projection_dim]
        
        # Normalize embeddings
        query_tokens_attended = query_tokens / query_tokens.norm(p=2, dim=-1, keepdim=True)
        
        # Cosine similarity logits
        logits_query = torch.matmul(query_tokens_attended, image_features[:, 0, :].unsqueeze(-1)).squeeze(-1)

        # Binary classification logits
        logits_binary_classification = self.binary_classifier(query_tokens_attended.squeeze(0))



        # ---- Obtain original CLIP outputs ----
        # Get the CLIP embeddings from the original model
        # text_inputs should be in the correct format, i.e., {'input_ids': ..., 'attention_mask': ...}
        clip_image_embeds = self.clip_model.get_image_features(pixel_values=image_inputs)
        clip_text_embeds = self.clip_model.get_text_features(**text_inputs)

        # Normalize the CLIP embeddings (CLIP does this internally as well)
        clip_image_embeds = clip_image_embeds / clip_image_embeds.norm(p=2, dim=-1, keepdim=True)
        clip_text_embeds = clip_text_embeds / clip_text_embeds.norm(p=2, dim=-1, keepdim=True)

        # Compute CLIP similarity scores
        logit_scale = self.clip_model.logit_scale.exp()
        # Shape: [batch, batch], assuming one-to-one pairing of images and texts
        clip_scores = logit_scale * torch.matmul(clip_image_embeds, clip_text_embeds.t())

        return logits_query, logits_binary_classification, clip_scores


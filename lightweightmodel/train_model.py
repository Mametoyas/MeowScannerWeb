import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor, ViTForImageClassification, Trainer, TrainingArguments
from PIL import Image
import os
from sklearn.preprocessing import LabelEncoder
import json

class CatBreedDataset(Dataset):
    def __init__(self, data_dir, processor, label_encoder=None):
        self.data_dir = data_dir
        self.processor = processor
        self.images = []
        self.labels = []
        
        # Load images and labels from folder structure
        for breed_folder in os.listdir(data_dir):
            breed_path = os.path.join(data_dir, breed_folder)
            if os.path.isdir(breed_path):
                for img_file in os.listdir(breed_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.images.append(os.path.join(breed_path, img_file))
                        self.labels.append(breed_folder)
        
        # Encode labels
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        else:
            self.label_encoder = label_encoder
            self.encoded_labels = self.label_encoder.transform(self.labels)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(),
            'labels': torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        }

class CatBreedTrainer:
    def __init__(self, train_dir, val_dir=None, model_name='google/vit-base-patch16-224'):
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        
        # Create datasets
        self.train_dataset = CatBreedDataset(train_dir, self.processor)
        self.val_dataset = CatBreedDataset(val_dir, self.processor, self.train_dataset.label_encoder) if val_dir else None
        
        # Setup model
        self.num_labels = len(self.train_dataset.label_encoder.classes_)
        self.model = ViTForImageClassification.from_pretrained(
            model_name, 
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True
        )
        
        # Save label mapping
        self.label_mapping = {i: label for i, label in enumerate(self.train_dataset.label_encoder.classes_)}
        
    def train(self, output_dir='./cat_breed_model', epochs=3, batch_size=8, learning_rate=2e-5):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="steps" if self.val_dataset else "no",
            eval_steps=500 if self.val_dataset else None,
            save_total_limit=4,
            remove_unused_columns=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
        )
        
        trainer.train()
        trainer.save_model()
        
        # Save label mapping
        with open(os.path.join(output_dir, 'label_mapping.json'), 'w') as f:
            json.dump(self.label_mapping, f)
        
        print(f"Model saved to {output_dir}")
        return trainer

# Usage example
if __name__ == "__main__":

    train_dir = r"S:\SE4AI\Cat-5\train"  # Update path
    val_dir = r"S:\SE4AI\Cat-5\val"      # Update path (optional)
    
    trainer = CatBreedTrainer(train_dir, val_dir)
    trainer.train(epochs=5, batch_size=4)
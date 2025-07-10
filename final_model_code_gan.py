import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from torch.nn.utils import spectral_norm
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import save_image, make_grid
from torchvision.transforms import InterpolationMode
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# CUDA initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    torch.cuda.set_device(0)
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()

# Data loading with balanced dataset
real_faces_path = "dataset/real"
fake_faces_path = "fake_faces/fake"

real_images = [os.path.join(real_faces_path, fname) for fname in os.listdir(real_faces_path) if fname.endswith(('.png', '.jpg', '.jpeg'))]
fake_images = [os.path.join(fake_faces_path, fname) for fname in os.listdir(fake_faces_path) if fname.endswith(('.png', '.jpg', '.jpeg'))]

min_size = min(len(real_images), len(fake_images))
real_images = real_images[:min_size]
fake_images = fake_images[:min_size]
all_images = real_images + fake_images
labels = [1] * min_size + [0] * min_size

print(f"Balanced dataset - Total images: {len(all_images)} (Real: {len(real_images)}, Fake: {len(fake_images)})")

# GAN-specific transforms
gan_transform = transforms.Compose([
    transforms.Resize((128, 128), interpolation=InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

class ImageDataset(Dataset):
    def __init__(self, file_paths, labels=None, transform=None, gan_mode=False):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.gan_mode = gan_mode

    def __len__(self): 
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            with Image.open(self.file_paths[idx]) as img:
                img = img.convert('RGB')
                if self.transform:
                    img = self.transform(img)
            
            if self.gan_mode:
                return img
            return img, torch.tensor(self.labels[idx], dtype=torch.float32)
        except Exception as e:
            print(f"Error loading image {self.file_paths[idx]}: {e}")
            placeholder = torch.randn(3, 128, 128) * 0.1 + 0.5
            if self.transform:
                placeholder = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(placeholder)
            if self.gan_mode:
                return placeholder
            return placeholder, torch.tensor(self.labels[idx], dtype=torch.float32)

# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Initial block
        self.init_block = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True)
        )
        
        # Main generator blocks
        self.blocks = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 1, kernel_size=1)),
            nn.Sigmoid()
        )
        
    def forward(self, z):
        z = z.view(-1, self.latent_dim, 1, 1)
        x = self.init_block(z)
        x = self.blocks(x)
        return x

# Discriminator Network (based on your EnhancedClassifier)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Main discriminator blocks
        self.blocks = nn.Sequential(
            # 128x128 -> 64x64
            spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 -> 32x32
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            spectral_norm(nn.Conv2d(512, 1024, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 1, kernel_size=1)),
            nn.Sigmoid()
        )
        
        # Final classification layer
        self.fc = spectral_norm(nn.Linear(1024*4*4, 1))
        
    def forward(self, x):
        # Get features
        features = self.blocks(x)
        
        # Apply attention at intermediate layer
        attn_features = self.attention(features[:, :256, :, :])
        features = features * attn_features
        
        # Flatten and classify
        features = features.view(features.size(0), -1)
        return self.fc(features)

# GAN Trainer
class GANTrainer:
    def __init__(self, generator, discriminator, device, latent_dim=100):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        self.latent_dim = latent_dim
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizers
        self.optimizer_G = optim.AdamW(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.AdamW(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Learning rate schedulers
        self.scheduler_G = ReduceLROnPlateau(self.optimizer_G, mode='min', patience=3)
        self.scheduler_D = ReduceLROnPlateau(self.optimizer_D, mode='min', patience=3)
        
        # GradScaler for mixed precision
        self.scaler_G = GradScaler()
        self.scaler_D = GradScaler()
        
        # Training history
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'd_real_acc': [],
            'd_fake_acc': [],
        }
        
    def train_epoch(self, dataloader, epoch):
        self.generator.train()
        self.discriminator.train()
        
        running_g_loss = 0.0
        running_d_loss = 0.0
        running_d_real_acc = 0.0
        running_d_fake_acc = 0.0
        
        for i, real_imgs in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(self.device)
            
            # Adversarial ground truths
            real_labels = torch.ones(batch_size, 1, device=self.device)
            fake_labels = torch.zeros(batch_size, 1, device=self.device)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            self.optimizer_D.zero_grad()
            
            # Real images
            with autocast():
                real_outputs = self.discriminator(real_imgs)
                d_loss_real = self.criterion(real_outputs, real_labels)
                d_real_acc = (torch.sigmoid(real_outputs) > 0.5).float().mean()
            
            # Fake images
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            with autocast():
                fake_imgs = self.generator(z)
                fake_outputs = self.discriminator(fake_imgs.detach())
                d_loss_fake = self.criterion(fake_outputs, fake_labels)
                d_fake_acc = (torch.sigmoid(fake_outputs) < 0.5).float().mean()
            
            # Total discriminator loss
            d_loss = (d_loss_real + d_loss_fake) / 2
            
            # Backward and optimize
            self.scaler_D.scale(d_loss).backward()
            self.scaler_D.step(self.optimizer_D)
            self.scaler_D.update()
            
            # -----------------
            #  Train Generator
            # -----------------
            
            self.optimizer_G.zero_grad()
            
            # Generate fake images
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            with autocast():
                fake_imgs = self.generator(z)
                outputs = self.discriminator(fake_imgs)
                g_loss = self.criterion(outputs, real_labels)  # Trick the discriminator
            
            # Backward and optimize
            self.scaler_G.scale(g_loss).backward()
            self.scaler_G.step(self.optimizer_G)
            self.scaler_G.update()
            
            # Update running metrics
            running_g_loss += g_loss.item()
            running_d_loss += d_loss.item()
            running_d_real_acc += d_real_acc.item()
            running_d_fake_acc += d_fake_acc.item()
            
            # Print progress
            if i % 50 == 0:
                print(f"[Epoch {epoch}] [Batch {i}/{len(dataloader)}] "
                      f"D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}, "
                      f"D acc: real {d_real_acc.item():.2f}, fake {d_fake_acc.item():.2f}")
        
        # Update learning rates
        self.scheduler_G.step(running_g_loss / len(dataloader))
        self.scheduler_D.step(running_d_loss / len(dataloader))
        
        # Save epoch metrics
        epoch_g_loss = running_g_loss / len(dataloader)
        epoch_d_loss = running_d_loss / len(dataloader)
        epoch_d_real_acc = 100 * running_d_real_acc / len(dataloader)
        epoch_d_fake_acc = 100 * running_d_fake_acc / len(dataloader)
        
        self.history['g_loss'].append(epoch_g_loss)
        self.history['d_loss'].append(epoch_d_loss)
        self.history['d_real_acc'].append(epoch_d_real_acc)
        self.history['d_fake_acc'].append(epoch_d_fake_acc)
        
        return epoch_g_loss, epoch_d_loss, epoch_d_real_acc, epoch_d_fake_acc
    
    def save_samples(self, epoch, n_samples=16):
        """Save generated samples for visualization"""
        z = torch.randn(n_samples, self.latent_dim, device=self.device)
        with torch.no_grad():
            samples = self.generator(z).cpu()
        
        # Denormalize
        samples = samples * 0.5 + 0.5
        
        # Create grid
        grid = make_grid(samples, nrow=4, padding=2, normalize=False)
        
        # Save image
        save_path = f'checkpoints/gan_samples_epoch_{epoch}.png'
        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.title(f'Generated Samples - Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_checkpoint(self, epoch):
        """Save model checkpoints"""
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'history': self.history,
        }, f'checkpoints/gan_checkpoint_epoch_{epoch}.pth')
    
    def plot_training_metrics(self):
        """Plot training metrics"""
        plt.figure(figsize=(15, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.history['g_loss'], label='Generator Loss')
        plt.plot(self.history['d_loss'], label='Discriminator Loss')
        plt.title('Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(self.history['d_real_acc'], label='Real Accuracy')
        plt.plot(self.history['d_fake_acc'], label='Fake Accuracy')
        plt.title('Discriminator Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gan_training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

def train_gan(train_indices, test_indices):
    # Create datasets
    train_dataset = ImageDataset(
        [all_images[i] for i in train_indices], 
        transform=gan_transform,
        gan_mode=True
    )
    
    test_dataset = ImageDataset(
        [all_images[i] for i in test_indices], 
        transform=gan_transform,
        gan_mode=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32,
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    print(f"Training: {len(train_dataset)}, Testing: {len(test_dataset)}")
    
    # Initialize GAN components
    generator = Generator(latent_dim=100)
    discriminator = Discriminator()
    
    # Create GAN trainer
    gan_trainer = GANTrainer(generator, discriminator, device)
    
    # Training parameters
    num_epochs = 75
    sample_interval = 5
    checkpoint_interval = 10
    
    # Create directory for checkpoints
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        g_loss, d_loss, d_real_acc, d_fake_acc = gan_trainer.train_epoch(train_loader, epoch + 1)
        
        print(f"[Epoch {epoch + 1}/{num_epochs}] "
              f"G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}, "
              f"D Acc: Real {d_real_acc:.2f}%, Fake {d_fake_acc:.2f}%")
        
        # Save generated samples
        if (epoch + 1) % sample_interval == 0:
            gan_trainer.save_samples(epoch + 1)
        
        # Save model checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            gan_trainer.save_checkpoint(epoch + 1)
    
    # Save final model
    gan_trainer.save_checkpoint('final')
    
    # Plot training metrics
    gan_trainer.plot_training_metrics()
    
    return gan_trainer

def evaluate_gan_discriminator(discriminator, dataloader):
    """Evaluate the discriminator as a classifier"""
    discriminator.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)
            
            outputs = discriminator(inputs)
            probs = torch.sigmoid(outputs)
            
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(probs.cpu().numpy())
    
    # Calculate metrics
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
    accuracy = 100 * np.mean(y_pred_binary == y_true)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Discriminator ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('gan_discriminator_roc.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Discriminator Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('gan_discriminator_cm.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Classification report
    report = classification_report(y_true, y_pred_binary, target_names=['Fake', 'Real'])
    print("\nDiscriminator Classification Report:")
    print(report)
    
    return accuracy, roc_auc

# Main execution
if __name__ == "__main__":
    # Create directory for checkpoints
    os.makedirs('checkpoints', exist_ok=True)
    
    # Train/test split (70:30)
    train_size = int(0.7 * len(all_images))
    
    # Get indices and shuffle them
    indices = list(range(len(all_images)))
    random.shuffle(indices)
    
    # Split while preserving class distribution
    real_indices = indices[:min_size]
    fake_indices = indices[min_size:]
    
    train_real_size = int(0.7 * len(real_indices))
    train_fake_size = int(0.7 * len(fake_indices))
    
    train_real_indices = real_indices[:train_real_size]
    test_real_indices = real_indices[train_real_size:]
    
    train_fake_indices = fake_indices[:train_fake_size]
    test_fake_indices = fake_indices[train_fake_size:]
    
    # Combine while maintaining stratification
    train_indices = train_real_indices + train_fake_indices
    test_indices = test_real_indices + test_fake_indices
    
    # Shuffle the combined indices
    random.shuffle(train_indices)
    random.shuffle(test_indices)
    
    print(f"\n{'='*80}")
    print(f"Training GAN on 70/30 Split")
    print(f"{'='*80}")
    
    # Train the GAN
    gan_trainer = train_gan(train_indices, test_indices)
    
    # Create test dataset with labels for discriminator evaluation
    test_dataset_with_labels = ImageDataset(
        [all_images[i] for i in test_indices], 
        [labels[i] for i in test_indices], 
        transform=gan_transform
    )
    
    test_loader_with_labels = DataLoader(
        test_dataset_with_labels, 
        batch_size=32,
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # Evaluate the discriminator
    print("\nEvaluating Discriminator on Test Set...")
    accuracy, auc_score = evaluate_gan_discriminator(gan_trainer.discriminator, test_loader_with_labels)
    
    print(f"\nFinal Discriminator Results:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"AUC: {auc_score:.4f}")
    
    print("\nGAN training completed successfully!")
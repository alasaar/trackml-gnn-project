import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import GradScaler, autocast

def train_model(model, train_graphs, device, epochs=50, lr=1e-3, save_path='trackml_gnn_weights.pth'):
    loader = DataLoader(train_graphs, batch_size=1, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pos_frac = train_graphs[0].y.mean().item()
    computed_pos_weight = (1 - pos_frac) / pos_frac
    pos_weight = torch.tensor([computed_pos_weight]).to(device)
    print(f"Positive fraction: {pos_frac:.4f} | Using pos_weight: {computed_pos_weight:.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Handle CPU vs GPU for AMP scaling
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    scaler = GradScaler(device_type) if device_type == 'cuda' else None
    
    scheduler = OneCycleLR(optimizer, max_lr=5e-3, steps_per_epoch=len(loader), epochs=epochs)

    model.train()
    print("Initiating Training Phase with Message Passing...")

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            if device_type == 'cuda':
                with autocast(device_type):
                    predictions = model(batch)
                    loss = criterion(predictions, batch.y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = model(batch)
                loss = criterion(predictions, batch.y)
                loss.backward()
                optimizer.step()
                
            scheduler.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {epoch_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    torch.save(model.state_dict(), save_path)
    print(f"[SUCCESS] True GNN Weights secured to {save_path}")
    return model
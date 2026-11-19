from model import forward_prop, w, b
from optimizer import sgd
from loss import sqr_loss
from synthetic_data import train_loader,val_loader
import tensorflow as tf

def evaluate(val_loader, w, b):
    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in val_loader:
        y_hat = forward_prop(x_batch, w, b)
        loss = sqr_loss(y_hat, y_batch)

        batch_size = x_batch.shape[0]

        total_loss += loss.numpy() * batch_size
        total_samples += batch_size

    return total_loss / total_samples
    

lr = 0.03
num_epochs = 10
train_history = []
val_history = []

for epoch in range(num_epochs):
    total_loss = 0.0
    total_samples = 0
    for x_batch, y_batch in train_loader:
        with tf.GradientTape() as tape:
            y_hat = forward_prop(x_batch, w,b)
            loss = sqr_loss(y_hat,y_batch)
        gradients = tape.gradient(loss,[w,b])
        sgd([w,b], gradients , lr)
        batch_size = x_batch.shape[0]

    total_loss += loss.numpy() * batch_size
    total_samples += batch_size
    train_loss = total_loss / total_samples
    
    # Validation
    val_loss = evaluate(val_loader, w, b)

    # Record losses
    train_history.append(train_loss)
    val_history.append(val_loss)
    print(
        f"Epoch {epoch + 1}, "
        f"Train Loss: {train_loss:.6f}, "
        f"Val Loss: {val_loss:.6f}"
    )

print("Learned weights:", w.numpy())
print("Learned bias:", b.numpy())
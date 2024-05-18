from model import Model, Model2
from torch import nn, optim 
import torch 
from tqdm.auto import tqdm
from dataset import train_loader, test_loader
import matplotlib.pyplot as plt

def plot_result(train_losses, test_losses, train_accs, test_accs):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, test_accs, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train and Test Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Loss_And_Acc.png')
    plt.show()


def main():
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model2().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.00001)
    num_epochs = 30
    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss, test_loss, train_acc, test_acc = .0, .0, .0, .0
        for X_train, y_train in tqdm(train_loader):
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            optimizer.zero_grad()
            y_logits = model(X_train)
            y_preds = y_logits.argmax(dim=1)
            tr_loss = loss_fn(y_logits, y_train)
            train_loss += tr_loss.item()
            train_acc += (y_preds == y_train).sum().item() / len(y_train)
            tr_loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            for X_test, y_test in tqdm(test_loader):
                X_test = X_test.to(device)
                y_test = y_test.to(device)
                y_logits = model(X_test)
                t_loss = loss_fn(y_logits, y_test)
                y_preds = y_logits.argmax(dim=1)
                test_loss += t_loss.item()
                test_acc += (y_preds == y_test).sum().item() / len(y_test)

        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        train_acc /= len(train_loader)
        test_acc /= len(test_loader)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f'Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%')
        
    torch.save(model.state_dict(), 'model.pt')
    plot_result(train_losses, test_losses, train_accs, test_accs)
        
if __name__ == '__main__':
    main()
from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os
import time
from models.data_setup import *
from models.spiking_model import*
import argparse
import torch
import numpy as np
import random

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"✓ Random seed set to {seed}")


def train_model(args):
    set_seed(args.seed)
    device = models.data_setup.get_device()

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_dataset = models.data_setup.get_train_dataset(data_path="./raw/")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=100,
        shuffle=True,
        num_workers=0, 
        generator=generator
    )

    test_set = models.data_setup.get_test_dataset(data_path="./raw/") 
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=100,
        shuffle=False,  # Don't shuffle test set
        num_workers=0,
        generator=generator
    )

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    acc_record = list([])
    loss_train_record = list([])
    loss_test_record = list([])


    snn = Event_SMLP_Fast(layers=args.layers,decay=args.decay)
    # snn.decay = args.decay
    snn.to(device)

    dims = []
    for name, layer in snn.named_modules():
        if isinstance(layer, nn.Linear):
           dims.append(layer.out_features)
    print(f"784->{'->'.join(map(str, dims))}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)

    snn.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)

    print(f"Current decay for training: {snn.decay}")
    test_string = "blabla"
    test_string += f"{snn.decay}"
    print(f"This is the test_string: {test_string}")
    for epoch in range(num_epochs):
        running_loss = 0
        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            snn.zero_grad()
            optimizer.zero_grad()

            images = images.float().to(device)
            outputs = snn(images)
            labels_ = torch.zeros(batch_size, 10).scatter_(1, labels.view(-1, 1), 1)
            labels_ =  labels_.to(device)
            loss = criterion(outputs, labels_)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (i+1)%100 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                        %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size,running_loss ))
                running_loss = 0
                print('Time elasped:', time.time()-start_time)
        correct = 0
        total = 0
        snn.eval()
        optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                targets = targets.to(device, dtype=torch.long)
                bsz = targets.size(0)
                optimizer.zero_grad()
                outputs = snn(inputs) 
                outputs = outputs.to(device)
                labels_ = torch.zeros(bsz, 10, device=device, dtype=outputs.dtype) 
                labels_.scatter_(1, targets.view(-1, 1), 1)  
                labels_ =  labels_.to(device)
                loss = criterion(outputs, labels_)
                _, predicted = outputs.max(1) 
                total += float(targets.size(0))
                correct += float(predicted.eq(targets).sum().item())
                if batch_idx %100 ==0:
                    acc = 100. * float(correct) / float(total)
                    print(batch_idx, len(test_loader),' Acc: %.5f' % acc)

        print('Iters:', epoch,'\n\n\n')
        print(f"{snn.decay}")
        print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
        acc = 100. * float(correct) / float(total)
        acc_record.append(acc)

        # Save ONLY if this is the best accuracy so far
        if acc > best_acc:
            print(f'New best accuracy: {acc:.2f}% (previous: {best_acc:.2f}%)')
            print('Saving best checkpoint...')
            best_acc = acc
            state = {
                'net': snn.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'acc_record': acc_record,
                'seed': args.seed,  # Save seed for reproducibility
                'args': vars(args)
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            layer_string = "_".join(str(x) for x in args.layers)
            layer_string += f"_{snn.decay}"
            torch.save(state, f"./checkpoint/ckpt_"+layer_string+".t7")

    print(f'\nTraining complete!')
    print(f'Best accuracy: {best_acc:.2f}%')
    print(f'Final accuracy: {acc:.2f}%')
    print(f'Random seed used: {args.seed}')

def main():
    parser = argparse.ArgumentParser(description="Run training for a specific Model Architecture")
    parser.add_argument("--layers", type=int, nargs="+", 
                       default=[784, 400, 10],
                       help="Layer sizes (e.g., --layers 784 400 10)")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed for reproducibility")
    parser.add_argument("--decay", type=float, default=0.2)
    parser.add_argument("--epoch", type=int, default=100, help="Specifiy the number of epochs") # NOT IMPLEMENTED
    parser.add_argument("--output", type=str, default="ckpt_spiking_model.t7", help="Location where checkpoint will be saved") #NOT IMPLEMENTED


    args = parser.parse_args()

    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Architecture: {args.layers}")
    print(f"Random seed: {args.seed}")
    print(f"Epochs: {args.epoch}")
    print("=" * 60)
    print()

    result = train_model(args)


if __name__ == '__main__':
    main()
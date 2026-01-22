def Run_Classification(IMG_SIZE,TRAIN_DIR,VAL_DIR,WORKERS,BATCH_SIZE,model,EPOCHS,DEVICE,optimizer,criterion,OUTPUT_DIR,
                       scheduler):
    import torch
    from tqdm import tqdm
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from sklearn.metrics import classification_report, f1_score, accuracy_score, roc_curve, auc, precision_recall_curve, \
        confusion_matrix
    from sklearn.preprocessing import label_binarize
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import copy
    import numpy as np
    from itertools import cycle

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.4768168032169342, 0.5012436509132385, 0.4276201128959656], [0.17313140630722046, 0.1456347107887268, 0.19238317012786865])
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.4768168032169342, 0.5012436509132385, 0.4276201128959656], [0.17313140630722046, 0.1456347107887268, 0.19238317012786865])
        ]),
    }

    #if not os.path.exists(VAL_DIR):
    #    print(f" Warning: Validation folder not found. Using Train folder for metrics.")
    #    VAL_DIR = TRAIN_DIR

    image_datasets = {
        'train': datasets.ImageFolder(TRAIN_DIR, data_transforms['train']),
        'val': datasets.ImageFolder(VAL_DIR, data_transforms['val'])
    }

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS) for x in
                   ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)

    print(f"✅ Dataset Loaded: {dataset_sizes['train']} training images.")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    print("\n--- Starting Advanced Training ---")

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Wrap the dataloader with tqdm
            # desc: adds a label to the bar
            # leave: clears the bar after the phase finishes
            pbar = tqdm(dataloaders[phase],
                        desc=f"{phase.capitalize()} Phase",
                        unit="batch",
                        leave=False)

            for inputs, labels in pbar:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Update running metrics
                current_loss = loss.item()
                running_loss += current_loss * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Update the progress bar with current batch loss
                pbar.set_postfix(loss=f"{current_loss:.4f}")
                #running_loss += loss.item() * inputs.size(0)
                #running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'  {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'baseline_best.pth'))

                scheduler.step(epoch_acc)

    print(f'\nBest Val Acc: {best_acc:.4f}')

    # --- 6. ADVANCED EVALUATION METRICS ---
    print("\n📊 Generating Comprehensive Report...")
    model.load_state_dict(best_model_wts)
    model.eval()

    y_true = []
    y_pred = []
    y_score = []  # For ROC/PR curves

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)

            # Get probabilities (Softmax) for ROC/PR
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(probs.cpu().numpy())

    # NEW CODE (TTA Prediction)
    #with torch.no_grad():
    #    for inputs, labels in dataloaders['val']:
    #        inputs = inputs.to(DEVICE)
    #        labels = labels.to(DEVICE)
    #
    #        # CALL TTA FUNCTION HERE
    #        # This returns the averaged probabilities directly (already softmaxed)
    #        probs = predict_tta(model, inputs)
    #
    #        # Get predicted class index
    #        _, preds = torch.max(probs, 1)
    #
    #        y_true.extend(labels.cpu().numpy())
    #        y_pred.extend(preds.cpu().numpy())
    #        y_score.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    # A. Standard Metrics
    print(classification_report(y_true, y_pred, target_names=class_names))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1-Score: {f1_score(y_true, y_pred, average='macro'):.4f}")

    # B. Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')  # annot=False because 38 classes is too crowded
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

    # C. ROC Curve (One-vs-Rest)
    # We binarize the labels to calculate ROC for multi-class
    y_test_bin = label_binarize(y_true, classes=range(num_classes))
    n_classes = y_test_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plotting specific ROC for Minority Class vs Average
    target_idx = dataloaders['val'].dataset.class_to_idx['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot']
    plt.figure()
    plt.plot(fpr[target_idx], tpr[target_idx], label=f'ROC curve (area = {roc_auc[target_idx]:.2f}) for Gray Leaf Spot')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Minority Class')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve_minority.png"))
    plt.close()

    # D. Precision-Recall Curve (Crucial for Imbalanced Data)
    precision = dict()
    recall = dict()
    pr_auc = dict()

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

    plt.figure()
    plt.plot(recall[target_idx], precision[target_idx],
             label=f'PR curve (area = {pr_auc[target_idx]:.2f}) for Gray Leaf Spot')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Minority Class')
    plt.legend(loc="best")
    plt.savefig(os.path.join(OUTPUT_DIR, "pr_curve_minority.png"))
    plt.close()

    print(f"\n✅ All results and plots saved to '{OUTPUT_DIR}'")

    epochs_range = range(1, EPOCHS + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss_history, label='Train Loss', color='blue', marker='o')
    plt.plot(epochs_range, val_loss_history, label='Val Loss', color='red', marker='x')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    # Note: Ensure you add train_acc_history to your loop logic similarly to val_acc_history
    plt.plot(epochs_range, train_acc_history, label='Train Acc', color='blue', marker='o')
    plt.plot(epochs_range, val_acc_history, label='Val Acc', color='red', marker='x')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "learning_curves.png"))
    plt.show()
    plt.close()
import cv2
import timm
import torch
import pytorch_lightning as pl
from torchvision import transforms

class TrafficSignClassifier(pl.LightningModule):
    def __init__(self, config, model, criterion, optimizer):
        super().__init__()

        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.save_hyperparameters(ignore=["model", "criterion", "optimizer"])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        preds = logits.argmax(dim=1)
        return preds

    def configure_optimizers(self):
        return self.optimizer

def create_model(model_name, num_classes, dropout, freeze_backbone, unfreeze_last_n=1):
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        drop_rate=dropout
    )

    if freeze_backbone:
        # Freeze everything
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the layers from the end
        # If n=1, it unfreezes the head. If n=4, it unfreezes the head + last 3 blocks.
        layers = list(model.children())
        for layer in layers[-unfreeze_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True

    return model

def create_classifier():
    ckpt_path = "services/models/checkpoints/efficientnet.ckpt"
    num_classes = 43
    model_template = create_model(
        model_name="efficientnet_b0",
        num_classes=num_classes,
        dropout=0.3,
        freeze_backbone=True,
        unfreeze_last_n=6
    )

    trainable_params = [p for p in model_template.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(trainable_params, lr=3e-4, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = TrafficSignClassifier.load_from_checkpoint(
        ckpt_path,
        num_classes=num_classes,
        model=model_template,
        optimizer=optimizer,
        criterion=criterion,
    )

    return classifier

def transform_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor = transform(image_rgb)
    tensor = tensor.unsqueeze(0)

    return tensor

def predict(classifier, image):
    with torch.no_grad():
        logits = classifier(image)

        predicted_class = logits.argmax(dim=1).item()

        return predicted_class

def get_class_label(class_number):
    CLASS_LABELS = {
        0: "Speed Limit 20 km/h",
        1: "Speed Limit 30 km/h",
        2: "Speed Limit 50 km/h",
        3: "Speed Limit 60 km/h",
        4: "Speed Limit 70 km/h",
        5: "Speed Limit 80 km/h",
        6: "End of Speed Limit 80 km/h",
        7: "Speed Limit 100 km/h",
        8: "Speed Limit 120 km/h",
        9: "No Overtaking",
        10: "No Overtaking for Trucks",
        11: "Priority at Next Intersection",
        12: "Priority Road",
        13: "Yield",
        14: "Stop",
        15: "No Traffic in Both Directions",
        16: "No Trucks Allowed",
        17: "No Entry",
        18: "General Danger",
        19: "Dangerous Left Bend",
        20: "Dangerous Right Bend",
        21: "Double Bend",
        22: "Uneven Road",
        23: "Slippery Road",
        24: "Road Narrows Ahead",
        25: "Road Work",
        26: "Traffic Signals Ahead",
        27: "Pedestrian Crossing",
        28: "School Crossing",
        29: "Bicycle Crossing",
        30: "Snow or Ice Warning",
        31: "Wild Animals Crossing",
        32: "End of Restrictions",
        33: "Turn Right Only",
        34: "Turn Left Only",
        35: "Go Straight Only",
        36: "Go Straight or Right",
        37: "Go Straight or Left",
        38: "Keep Right",
        39: "Keep Left",
        40: "Roundabout",
        41: "End of No Overtaking",
        42: "End of No Overtaking for Trucks"
    }

    return CLASS_LABELS[class_number]
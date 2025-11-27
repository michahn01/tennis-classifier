# **ResNet50 + LSTM (Baseline)**

## **0. Dataset Information**
- p1 to p31 correspond to beginners, p32 to p55 to experts
- Classes to data/ directories:
  - **backhand** → backhand, backhand_slice, backhand2hands  
  - **forehand** → forehand_flat, forehand_openstands, forehand_slice
  - **service** → flat_service, kick_service, slice_service  
  - **bvolley** → backhand_volley  
  - **fvolley** → forehand_volley  
  - **smash** → smash

## **1. Preprocessing**
- Define **T**, the number of sampled frames per video  
  - Example: uniformly sample **T = 32** frames
- For each selected frame:
  - Resize to **H × W** (e.g., **224 × 224** for ResNet50)
  - Convert to tensor **(C, H, W)** where **C = 3**
  - Normalize using **ImageNet statistics**
  - (Optional) Data augmentation if helpful
- After preprocessing:
  - Each video → tensor shape **(T, 3, 224, 224)**

---

## **2. Feature Extraction Using ResNet-50**

### **Prepare ResNet-50**
- Load pretrained ResNet-50
- Remove final 1000-class FC layer
- Feature extractor:
  - Input: **(3, 224, 224)**
  - Output: **2048-D vector**

### **Applying Feature Extractor**
1. Input videos: **(B, T, C, H, W)**
2. Flatten frames:  
   - `all_frames = (B*T, C, H, W)`
3. Extract features:  
   - `all_features = (B*T, 2048)`
4. Reshape back:  
   - `frame_features = (B, T, 2048)`

---

## **3. LSTM Over Frame Features**

### **Define LSTM**
- Input size: **D = 2048**
- Hidden size: **H** (e.g., 256)
- Number of layers: **L** (1–2)
- Bidirectional: optional
- Dropout: use if **L > 1**

### **Input/Output**
- Input: **(B, T, D)**
- Output:  
  - `lstm_out`: **(B, T, H)**  
  - `h_n`: **(num_layers, B, H)**  
  - `c_n`: **(num_layers, B, H)**

### **Using the LSTM Output**
- For baseline: use **last hidden state** of the last layer  
  - `h_last_layer = h_n[-1]` → shape **(B, H)**
- Alternative: pooling (average, attention)

---

## **4. Classification Head (FC Layer)**

### **Purpose**
- Map video representation → class logits

### **Classes (6 total)**
- **backhand** → backhand, backhand_slice, backhand2hands  
- **forehand** → forehand_flat, forehand_openstands, forehand_slice
- **service** → flat_service, kick_service, slice_service  
- **bvolley** → backhand_volley  
- **fvolley** → forehand_volley  
- **smash** → smash

### **Fully Connected Mapping**
- `logits = W * h_last_layer + b`
  - logits: **(B, num_classes)**  
  - W: **(num_classes, H)**  
  - b: **(num_classes)**  
- Softmax used during training (CrossEntropyLoss handles this)

### **Forward Pass Summary**
- `(B, T, 3, 224, 224)`  
  → ResNet → `(B, T, 2048)`  
  → LSTM → `(B, H)`  
  → FC → `(B, num_classes)`

---

# **Training Loop**

## **Initialize Model Components**
- Load pretrained ResNet-50
- Remove final FC to create feature extractor
- Optionally freeze early ResNet layers
- Create LSTM + FC classifier

## **Loss and Optimization**
- Loss: **CrossEntropyLoss**
- Optimizer: **Adam** or **SGD**
- Learning rate:
  - ~1e-3 for scratch components
  - ~1e-4 if fine-tuning ResNet

## **Epoch Loop**
- For each mini-batch:
  - Load B videos + labels
  - Preprocess → `(B, T, 3, 224, 224)`
  - Forward pass → logits
  - Compute loss
  - Zero grads → `optimizer.zero_grad()`
  - Backprop → `loss.backward()`
  - Update → `optimizer.step()`

---

# **Validation**
- Run after each epoch
- Compute:
  - Accuracy
  - F1-score
  - Confusion matrix (optional)

# **Test**
- Evaluate final trained model on test dataset

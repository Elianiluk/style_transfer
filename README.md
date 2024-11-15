
# **Style Transfer Using Convolutional Neural Networks (CNN)**

## **Overview**
This program implements a style transfer technique that blends two images: a **content image** and a **style image**. The goal is to transfer the artistic style of one image onto the content of another image. 

### **How It Works**
1. **Content Features**: Extracted from a specific convolutional layer of the content image using a pre-trained CNN (VGG19).
2. **Style Features**: Extracted from multiple layers of the style image to capture textures and patterns.
3. **Optimization**: A target image is initialized as the content image and iteratively updated to minimize a combined loss function:
   - **Content Loss**: Measures the difference in content between the target and content image.
   - **Style Loss**: Measures the difference in style between the target and style image using Gram Matrices.

## **Key Features**
1. **Customizable Layers**:
   - Allows selection of convolutional layers for extracting content and style features.
2. **Gram Matrices**:
   - Captures the style of the style image by computing correlations between feature maps.
3. **Pre-Trained Model**:
   - Uses a pre-trained VGG19 model to extract image features.
4. **Intermediate Visualization**:
   - Displays the target image at regular intervals during the optimization process.

---

## **Usage**

### **Requirements**
Install the required libraries before running the code:
```bash
pip install torch torchvision matplotlib numpy pillow requests
```

### **File Structure**
- `style_transfer.py`: The main script containing the implementation.
- `images/`: Directory containing the content and style images.

### **Run the Program**
1. Place your content and style images in the `images/` directory.
2. Modify the file paths in the script:
   ```python
   content = load_image('images/content.jpg').to(device)
   style = load_image('images/style.jpg', shape=content.shape[-2:]).to(device)
   ```
3. Run the script:
   ```bash
   python style_transfer.py
   ```

---

## **Program Workflow**
1. **Image Loading and Preprocessing**:
   - Resizes and normalizes images for compatibility with VGG19.
   - Converts images into PyTorch tensors.

2. **Feature Extraction**:
   - Uses a pre-trained VGG19 model to extract features for both content and style images.

3. **Training**:
   - Iteratively updates the target image by minimizing the total loss (content + style loss).

4. **Visualization**:
   - Displays the intermediate and final target image.

---

## **Model Architecture**
The program uses a pre-trained VGG19 network, focusing on its convolutional layers to extract content and style features. Key layers:
- **Content Layer**: `conv4_2`
- **Style Layers**: `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1`

---

## **Customization**
1. **Change Content and Style Layers**:
   - Modify the layers used for feature extraction in the `get_features` function.

2. **Adjust Weights**:
   - Modify the weights for each style layer in `style_weights` to emphasize or de-emphasize specific patterns.

3. **Learning Rate and Steps**:
   - Adjust the learning rate and the number of optimization steps:
     ```python
     optimizer = optim.Adam([target], lr=0.003)
     steps = 5000
     ```

---

## **Results**
1. **Intermediate Images**:
   - Displays the target image at intervals during optimization to show progress.

2. **Final Image**:
   - Outputs a stylized version of the content image with the artistic style of the style image.

---

## **Contact**
**Author**: Elian Iluk  
**Email**: elian10119@gmail.com  

Feel free to reach out for any questions or feedback regarding the program.


import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

# import torch_directml
# device = torch_directml.device()
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


# model = models.vgg19(pretrained=True).features
# print(model)
# => Get layers: 0, 5, 10, 19, 28
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        
        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(pretrained=True).features[:29]
    
    def forward(self, x):
        features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features
    
def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)


image_size = 356

loader = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
)

original_img = load_image("lisa.jpg")
style_img = load_image("styles/style1.jpg")

model = VGG().to(device).eval() # Freeze weights

# generated = torch.randn(original_img.shape, device=device, requires_grad=True)
generated = original_img.clone().requires_grad_(True)

# Hyperparameters
total_steps = 6000
learning_rate = 0.001
alpha = 1 
beta = 0.01
optimizer = optim.Adam([generated], lr=learning_rate)

# for step in range(total_steps):
#     generated_features = model(generated)
#     original_img_features = model(original_img)
#     style_features = model(style_img)

#     style_loss = original_loss = 0

#     for gen_feature, orig_feature, style_feature in zip(generated_features, original_img_features, style_features):
#         batch_size, channel, height, width = gen_feature.shape
#         original_loss += torch.mean((gen_feature - orig_feature) ** 2)

#         # Compute Gram Matrix
#         G = gen_feature.view(channel, height * width).mm(
#             gen_feature.view(channel, height * width).t() # Transpose
#         )

#         A = style_feature.view(channel, height * width).mm(
#             style_feature.view(channel, height * width).t() # Transpose
#         )

#         style_loss += torch.mean((G - A)**2)

#     total_loss = alpha * original_loss + beta * style_loss
#     optimizer.zero_grad()
#     total_loss.backward()
#     optimizer.step()

#     if step % 50 == 0:
#         print(total_loss)
#         save_image(generated, f"generated_{step}.png")
lisa = load_image('lisa.jpg')
print(f'Lisa: {lisa.shape}')

me = load_image(r"C:\Users\USER\OneDrive\Pictures\Camera Roll\me.jpg")
print(f'Me: {me.shape}')
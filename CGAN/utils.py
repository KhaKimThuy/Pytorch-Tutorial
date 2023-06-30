import torch
import torch.nn as nn
"""
WGAN_GP
    Khi huấn luyện mô hình GAN truyền thống, discriminator có thể trở nên quá mạnh và khiến cho
    generator không đủ khả năng tạo ra các dữ liệu đa dạng và chất lượng, dẫn đến hiện tượng mode collapse.

    WGAN-GP giải quyết vấn đề này bằng cách chỉ sử dụng một phần ảnh thật và phần còn lại là các ảnh
    được sinh ra bởi generator để huấn luyện discriminator, thay vì sử dụng toàn bộ ảnh thật để huấn
    luyện. Điều này giúp mô hình WGAN-GP tránh được việc discriminator quá mạnh vì nó phải phân biệt
    giữa hai loại dữ liệu khác nhau trong quá trình huấn luyện.

    Đồng thời, thay vì sử dụng binary crossentropy loss function như trong GAN, WGAN-GP sử dụng
    Wasserstein distance để đo lường khoảng cách giữa phân phối của dữ liệu thật và dữ liệu giả.
    Wasserstein distance có tính chất liên tục, dẫn đến việc gradient của nó dễ dàng tính toán
    hơn so với binary crossentropy loss, từ đó giúp giảm thiểu hiện tượng vanishing gradient và
    mode collapse.
"""
def gradient_penalty(critic, labels, real, fake,  device='cpu'):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)

    # Calculates critic scores
    mixed_scores = critic(interpolated_images, labels)

    gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1)**2)
    return gradient_penalty
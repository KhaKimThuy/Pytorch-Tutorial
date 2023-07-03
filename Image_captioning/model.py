# import torch
# import torch.nn as nn
# import torchvision.models as models

# class EncoderCNN(nn.Module):
#     def __init__(self, embed_size, train_CNN=False):
#         super(EncoderCNN, self).__init__()
#         self.train_CNN = train_CNN
#         self.inception = models.inception_v3(pretrained=True, aux_logits=True)
#         self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, images):
#         features = self.inception(images)

#         for name, param in self.inception.named_parameters():
#             if "fc.weight" in name or "fc.bias" in name:
#                 param.requires_grad = True
#             else:
#                 param.requires_grad = self.train_CNN

#         return self.dropout(self.relu(features))
    
# '''`nn.Embedding`
# hữu ích trong việc biểu diễn các từ hoặc đối tượng trong không gian
# vector để sử dụng làm đầu vào cho các mô hình machine learning và deep learning.'''

# '''`num_layers`
# là một tham số được sử dụng để xác định số lớp LSTM được xếp chồng lên nhau.
# Việc xếp chồng các lớp LSTM cho phép mô hình học được biểu diễn các mức độ phức tạp của
# dữ liệu chuỗi dài hơn và mô hình hóa các phụ thuộc dài hạn trong dữ liệu chuỗi.'''
# class DecoderRNN(nn.Module):
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
#         super(DecoderRNN, self).__init__()
#         self.embed = nn.Embedding(vocab_size, embed_size)
#         self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
#         self.linear = nn.Linear(hidden_size, vocab_size)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, features, caption):
#         embedding = self.dropout(self.embed(caption))
#         embedding = torch.cat((features.unsqueeze(0), embedding), dim=0)
#         hiddens, _ = self.lstm(embedding)
#         output = self.linear(hiddens)
#         return output
    

# class CNN2RNN(nn.Module):
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
#         super(CNN2RNN, self).__init__()
#         self.encoderCNN = EncoderCNN(embed_size)
#         self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

#     def forward(self, image, caption):
#         features = self.encoderCNN(image)
#         output = self.decoderRNN(features, caption)
#         return output
    
#     def caption_image(self, image, vocabulary, max_length=50):
#         result_caption = []

#         with torch.no_grad():
#             x = self.encoderCNN(image).unsqueeze(0)
#             states = None

#             for _ in range(max_length):
#                 hiddens, states = self.decoderRNN.lstm(x, states)
#                 output = self.decoderRNN.linear(hiddens.squeeze(0))
#                 predicted = output.argmax(1)

#                 result_caption.append(predicted.item())
#                 x = self.decoderRNN.embed(predicted).unsqueeze(0)

#                 if vocabulary.itos[predicted.item()] == "<EOS>":
#                     break
            
#             return [vocabulary.itos[idx] for idx in result_caption]




import torch
import torch.nn as nn
import statistics
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(pretrained=True, aux_logits=True)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.times = []
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images)
        # for name, param in self.inception.named_parameters():
        #     if "fc.weight" in name or "fc.bias" in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = self.train_CNN
        return self.dropout(self.relu(features))


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs


class CNN2RNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNN2RNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]
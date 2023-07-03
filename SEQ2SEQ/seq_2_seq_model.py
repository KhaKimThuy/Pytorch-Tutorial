import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import spacy
import random
import nltk
from torch.utils.tensorboard import SummaryWriter

# Tải model ngôn ngữ đã được huấn luyện sẵn
# spacy_ger = spacy.load("de_core_news_sm")
# spacy_ger = spacy.load("de")
spacy_ger = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")
# spacy_de = spacy.load('de')
# spacy_eng = spacy.load('en')


# nlp = spacy.load("de_core_news_sm")
# exit()


# Hàm tạo tokenizer cho mô hình ngôn ngữ
def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

'''`Field`
    chứa các thông tin và quyết định liên quan đến định dạng dữ liệu,
    tiền xử lý và xử lý sau khi khi đã nạp dữ liệu.
'''
german = Field(tokenize=tokenize_ger, lower=True, init_token='<sos>', eos_token='<eos>')

english = Field(tokenize=tokenize_eng, lower=True, init_token='<sos>', eos_token='<eos>')

# Chia tập dữ liệu huấn luyện
train_data, validation_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                         fields = (german, english))

'''`build_vocab`
    là một phương thức được sử dụng để xây dựng từ điển (vocabulary) từ tập huấn
    luyện của dữ liệu văn bản. Từ điển này chứa toàn bộ các từ và từ điển của tập dữ liệu, cùng
    với các chỉ số tương ứng của chúng.
    `max_size`: Đây là tham số xác định số lượng tối đa các từ mà từ điển có thể chứa. Chỉ những
    từ phổ biến nhất (theo thứ tự tần suất) sẽ được bao gồm trong từ điển.
    `min_freq`: Đây là tham số xác định tần suất tối thiểu mà một từ cần phải xuất hiện trong tập
    huấn luyện để được bao gồm trong từ điển.
'''
german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

class Encoder(nn.Module):
    # input_size: kích thước của từ điển ngôn ngữ
    # embedding_size: Mỗi từ sẽ được ánh xạ đến không gian d chiều
    
    ''' `num_layers`
        xác định số lớp LSTM được xếp chồng lên nhau. Mỗi lớp LSTM là một lớp LSTM đầy đủ, nhận đầu vào
        từ lớp LSTM ở trên nó và xuất ra đầu ra cho lớp LSTM ở phía dưới. Mỗi lớp LSTM có thể tìm hiểu
        các mức độ tuần tự khác nhau của dữ liệu và tạo ra các đầu ra có kích thước nhỏ hơn. Điều này
        cho phép mô hình tập trung vào các đặc trưng và thông tin quan trọng hơn khi diễn giải dữ liệu.
    '''
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p_dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p_dropout)

        '''`embedding`
            nhận vào một số nguyên làm đầu vào, thường là một chỉ số đại diện cho từ hoặc
            thực thể, và trả về một vector tương ứng từ ma trận nhúng (biểu diễn từ dưới dạng các
            vector dựa trên ngữ nghĩa của chúng). Kích thước của ma trận nhúng sẽ
            phụ thuộc vào kích thước của từ điển và số chiều của vector nhúng.
        '''
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)
        self.fc_hidden = nn.Linear(hidden_size*2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, x):
        # x shape = (seq_length, batch_size)
        # embedding shape = (seq_length, batch_size, embedding_size)
        '''
            Mỗi từ trong câu sẽ được biểu diễn trong không gian embedding_size chiều
        '''
        embedding = self.dropout(self.embedding(x))

        # Context vector chứa trong 'hidden', 'cell'
        '''
        Mỗi layer chứa nhiều RNN unit (thông tin truyền từ unit này -> unit khác)
        Đầu vào của layer sau sẽ là hidden_state của mỗi từ trong layer đầu

        Hidden state: thể được xem như bộ nhớ ngắn hạn của mô hình. Nó được tính toán dựa trên đầu vào hiện tại và
        hidden state trước đó. Hidden state đóng vai trò quan trọng trong việc truyền thông tin từ một thời điểm
        sang thời điểm tiếp theo trong quá trình tuần tự.

        Cell state: được sử dụng để lưu trữ và truyền thông tin dài hạn giữa các thời điểm
        có khả năng giữ lại thông tin quan trọng và loại bỏ thông tin không cần thiết bằng cách sử dụng các cổng (gates), 
        cho phép mô hình học và theo dõi các phụ thuộc dài hạn trong dữ liệu.


        outputs: chứa tất cả các hidden state của mỗi từ trong layer cuối cùng
        hidden: chứa tất cả các hidden state cuối cùng của mỗi layer
        cell: chứa tất cả các cell state cuối cùng trong mỗi layer
        '''
        encoder_states, (hidden, cell) = self.rnn(embedding)

        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))


        '''
        encoder_states: chứa thông tin của mỗi từ
        hidden, cell: chứa thông tin tổng hợp của tất cả các từ
        '''
        return encoder_states, hidden, cell

class Decoder(nn.Module):
    # input_size = output_size
    # output: vector thể hiện xác suất của mỗi từ
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p_dropout):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        # self.rnn = nn.LSTM(hidden_size*2 + embedding_size, hidden_size, num_layers, dropout=p_dropout)
        self.rnn = nn.LSTM(hidden_size*2 + embedding_size, hidden_size, num_layers)

        self.energy = nn.Linear(hidden_size*3, 1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p_dropout)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()


    def forward(self, x, encoder_states, hidden, cell):
        '''
            Decoder sẽ dự đoán word by word
        '''
        # x shape: (batch_size) -> (1, batch_size) Mỗi batch 1 từ
        x = x.unsqueeze(0)
        
        # embedding shape: (1, batch_size, embedding_size)
        embedding = self.dropout(self.embedding(x))

        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)

        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))

        # (sequence_length, batch_size, 1)
        attention = self.softmax(energy)

        # (N, 1, sequence_length)
        attention = attention.permute(1, 2, 0)

        # (N, sequence_length, hidden_size*2)
        encoder_states = encoder_states.permute(1,0,2)

        context_vector = torch.bmm(attention, encoder_states).permute(1,0,2)

        rnn_input = torch.cat((context_vector, embedding), dim=2)

        # outputs shape: (1, batch_size, hidden_size)
        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        # predictions shape: (1, batch_size, vocab_size) 
        predictions = self.fc(outputs).squeeze(0)

        return predictions, hidden, cell



class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5, device='cpu'):
        '''
            source : câu muốn dịch
            target: ground truth (kết quả dịch của source)
            Trong Decoder kết quả dự đoán dịch của từ đầu tiên sẽ được làm đầu vào của unit rnn tiếp theo,
            và dự đoán này không phải luôn đúng -> ruin sentence
            (không nên để nó làm đầu vào cho unit tiếp theo). Do đó, nên dùng đầu ra ground truth để
            làm đầu vào cho các unit
            
            teacher_force_ratio = 0.5: 50% dùng predicted word - 50% dùng ground truth để làm input cho
            các unit (Không teacher_force_ratio = 1, vì trong test time chúng ta không có ground truth cho
            đầu vào của các unit RNN tiếp theo, do đó quá trình huấn luyện cũng cần mô phỏng điều này để đảm
            bảo khả năng hoạt động chính xác của mô hình trong thực tế và mô hình không có khả năng xử lý
            tốt các tình huống mà nó chưa được "hướng dẫn" bằng ground truth.)
        '''
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        '''
            Dự đoán mỗi lần 1 từ, với mỗi từ được dự đoán ta làm với toàn bộ batch, 
        '''
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        encoder_states, hidden, cell = self.encoder(source)

        # Grab start token
        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            
            x = target[t] if random.random() < teacher_force_ratio else best_guess
        
        return outputs


# repear, permute, bmm
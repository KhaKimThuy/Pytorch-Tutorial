from seq_2_seq_model import *
from utils import *
import torch_directml
# device = torch_directml.device()
device = "cuda" if torch.cuda.is_available() else "cpu"
load_model = False
save_model = True

# Training hyperparameters
num_epochs = 20
learning_rate = 0.001
batch_size = 64

# Model hyperparameters

input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 1
enc_dropout = 0.5
dec_dropout = 0.5

# Tensorboard
writer = SummaryWriter(f'runs/loss_plot')
step = 0
'''
    BucketIterator là một phương pháp giúp tổ chức lại dữ liệu thành các batch có độ dài tương đối gần nhau,
từ đó tối đa hoá sự tương tự về độ dài giữa các mẫu trong cùng một batch.
    Trước tiên phân tách các mẫu dữ liệu thành các bucket, trong đó mỗi bucket chứa các mẫu có độ dài tương
tự. Sau đó, các bucket này được sắp xếp theo độ dài và được chia thành các batch dựa trên số lượng mẫu
trong mỗi bucket và bất kỳ một tiêu chí khác. Mỗi batch chứa các mẫu có độ dài gần nhau, từ đó giúp cải
thiện hiệu suất huấn luyện.
'''
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x : len(x.src),
    device=device
)

encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)
decoder_net = Decoder(
    input_size_decoder,
    decoder_embedding_size,
    hidden_size,
    output_size,
    num_layers,
    dec_dropout,
).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
'''
 `'<pad>'`
    là một token đặc biệt nhằm đại diện cho việc padding (điền vào) các câu văn bản để có độ dài đồng nhất.
    , thường được sử dụng trong quá trình xử lý dữ liệu văn bản để tạo các batch có kích thước
    như nhau khi đưa vào mô hình.
'''
pad_idx = english.vocab.stoi['<pad>'] # để truy xuất chỉ số số hóa tương ứng với từ '<pad>'
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)


sentence = "    "
for epoch in range(num_epochs):
    print(f'Epoch {epoch} / {num_epochs}')
    
    checkpoint = {'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}
    save_checkpoint(checkpoint)

    model.eval()

    translated_sentence = translate_sentence(model, sentence, german, english, device, max_length=50)
    print(f"Translated example sentence: \n {translated_sentence}")
    model.train()

    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(inp_data, target)

        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        loss.backward()

        '''
        `torch.nn.utils.clip_grad_norm_`
        được sử dụng để giới hạn (clipping) giá trị của gradient trong quá trình huấn luyện mạng neural
        network. Việc giới hạn gradient có thể giúp ổn định quá trình lan truyền ngược (backpropagation)
        và ngăn chặn hiện tượng gradient exploding (biến mất gradient). Nếu giá trị của gradient vượt
        quá ngưỡng, gradient sẽ được scaling sao cho giá trị của nó không vượt quá ngưỡng được chỉ định.
        
        `max_norm` là một giá trị dương (positive value) đại diện cho mức độ giới hạn mà gradient không
        được vượt qua.
        '''
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        writer.add_scalar('Training loss', loss, global_step=step)
        step+=1

'''
Bleu score
là thước đo chất lượng của các mô hình máy dịch,
bleu score > 0.6 thường được xem là có chất lượng tốt
'''






# 사용된 코드 : https://wikidocs.net/63565
# 아래의 코드는 위의 링크에 있는 코드를 주석을 통해 해석해본 코드입니다.

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

# Device 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 랜덤 시드 고정
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# Parameter
learning_rate = 0.001
training_epochs = 30
batch_size = 100

# MNIST data를 받아 tensor로 저장하는 과정
mnist_train = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                          train=True, # True를 지정하면 훈련 데이터로 다운로드
                          transform=transforms.ToTensor(), # 텐서로 변환
                          download=True)
mnist_test = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                         train=False, # False를 지정하면 테스트 데이터로 다운로드
                         transform=transforms.ToTensor(), # 텐서로 변환
                         download=True)

# Tensor로 data loader를 만드는 과정
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        ### 함수 해석
        #   Conv2d(input channel, output channel, kernel size, stride, padding)
        #       input channel : input 2d data 개수
        #       output channel : output 2d data 개수
        #       kernel size : 필터의 크기
        #       stride : 필터 적용 간격
        #       padding : 외곽에 0으로 둘러싸는 두께
        #   ReLU()
        #   MaxPool2d(kernel size, stride)
        #       kernel size : 필터의 크기
        #       stride : 필터 적용 간격
        #   Linear(input dim, output dim)
        #       input dim : input 1d data 개수
        #       output dim : output 1d data 개수

        ### 첫번째 층 (Conv, ReLU, Pool)
        #   (28 * 28 * 1)
        #   -> (28 * 28 * 32) (Conv2d)
        #   -> (28 * 28 * 32) (ReLU)
        #   -> (14 * 14 * 32) (MaxPool2d)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        ### 두번째 층 (Conv, ReLU, Pool)
        #   (14 * 14 * 32)
        #   -> (14 * 14 * 64) (Conv2d)
        #   -> (14 * 14 * 64) (ReLU)
        #   -> (7 * 7 * 64) (MaxPool2d)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        ### 세번째 층 (FC)
        #   (3136) = (7 * 7 * 64)
        #   -> (10) (Linear)
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)

        # Linear 이후 후처리 작업 (weight 초기화)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    # CNN
    def forward(self, x):

        ### 첫번째 층 (Conv, ReLU, Pool)
        #   (28 * 28 * 1) -> (14 * 14 * 32)
        out = self.layer1(x)

        ### 두번째 층 (Conv, ReLU, Pool)
        #   (14 * 14 * 32) -> (7 * 7 * 64)
        out = self.layer2(out)

        ### 2D data에서 1D data로 변환
        #   (7 * 7 * 64) -> (3136)
        out = out.view(out.size(0), -1)

        ### 세번째 층 (FC)
        #   (3136) -> (10)
        out = self.fc(out)

        return out

### CNN Model을 Device에 연결
model = CNN().to(device)

# 손실함수 : 실제 값과 결과 값의 비교
criterion = torch.nn.CrossEntropyLoss().to(device)

# CNN model의 가중치를 갱신할 optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(data_loader)
print('총 배치의 수 : {}'.format(total_batch))

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:
        # data load
        X = X.to(device)
        Y = Y.to(device)

        # CNN 모델을 통해 비용 측정
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)

        # 변화도 누적 초기화
        optimizer.zero_grad()

        # 역전파
        cost.backward()

        # 모델 갱신 과정
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

# 학습하지 않을 때의 코드, 가중치 갱신 과정은 생략함
with torch.no_grad():

    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())
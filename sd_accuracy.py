import torchvision
from torchvision import transforms
import os
from torch.utils.data import Dataset,DataLoader
import torch

#바꾸는것 
# 1. PATH 에 숫자 
# 2. testset 에 샘플경로

# 모델경로
PATH = './content/scalp_weights/aram_model1.pt'   # aram_model6.pt 모델파일 이름 바꿔주기 
 
# Cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(PATH, map_location=device)

# 전처리-트랜스폼 규칙 선언 # model1_train 코드의 validation set 의 트랜스폼 규칙과 동일하게 함
transforms_test = transforms.Compose([
                                        transforms.Resize([int(600), int(600)], interpolation=transforms.InterpolationMode.BOX),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                      ])
# root 경로 폴더 속 jpg를 전처리, 텐서화 (rood 속에 폴더를 하나 더 만들어서 jpg를 묶어야 함)
testset = torchvision.datasets.ImageFolder(root = './content/sample/model1sample' ,
                    transform = transforms_test)

# DataLoader를 통해 네트워크에 올리기
from torch.utils.data import Dataset,DataLoader 
testloader = DataLoader(testset, batch_size=2, shuffle=False, num_workers=0)



## 아웃풋, 로스, 프레딕, 아큐러시

# output_list = []
model.eval() # 평가모드로 전환 # 평가모드와 학습모드의 layer 구성이 다르다
correct = 0

# 로스 연산을 위한
import torch.nn.functional as F   # F : 테스트_로스 연산 함수
test_loss = 0
from tqdm import tqdm # 진행률 표시를 위한

if __name__ == '__main__':
    with torch.no_grad(): # 평가할 땐  gradient를 backpropagation 하지 않기 때문에 no grad로 gradient 계산을 막아서 연산 속도를 높인다
            for data, target in tqdm(testloader):                                   
                data, target  = data.to(device), target.to(device) 
                output = model(data)   # model1에 데이터를 넣어서 아웃풋 > [a,b,c,d] 각 0,1,2,3 의 확률값 리턴 가장 큰 것이 pred
                # output_list.append(output);
                test_loss += F.nll_loss(output, target, reduction = 'sum').item()  # test_loss변수에 각 로스를 축적
                pred = output.argmax(dim=1, keepdim=True) # argmax : 리스트에서 최댓값의 인덱스를 뽑아줌 > y값아웃풋인덱
                correct += pred.eq(target.view_as(pred)).sum().item() # accuracy 측정을 위한 변수 # 각 예측이 맞았는지 틀렸는지 correct변수에 축적 맞을 때마다 +1  # # view_as() 함수는 target 텐서를 view_as() 함수 안에 들어가는 인수(pred)의 모양대로 다시 정렬한다. #  view_as() 함수는 target 텐서를 view_as() 함수 안에 들어가는 인수(pred)의 모양대로 다시 정렬한다. #  pred.eq(data) : pred와 data가 일치하는지 검사

test_loss /= len(testloader.dataset)  # 로스축적된 로스를 데이터 수(경로안jpg수)로 나누기

# 아큐러시 출력 ( :.4f 소수점반올림 )
# print('\nTest set Accuracy: {}/{} ({:.4f}%)\n'.format(correct, len(testloader.dataset), 100. * correct / len(testloader.dataset)))  # 축적된 예측값을 데이터 개수로 나누기 *100 > 확률%값

# 로스, 아큐러시 출력
print('\nTest set: Average Loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(test_loss, correct, len(testloader.dataset), 100. * correct / len(testloader.dataset)))
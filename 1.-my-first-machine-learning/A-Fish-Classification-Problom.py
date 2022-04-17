import matplotlib.pyplot as plt
import torch

bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

length = bream_length + smelt_length
weight = bream_weight + smelt_weight

fish_data = torch.tensor([
    [l, w] for l, w in zip(length, weight) 
])

# 도미와 생선을 맞출 수 있도록 정답 데이터를 만들어 준다.
# 정답 데이터는 위 데이터의 row의 길이를 가져야 한다.

bream_target = torch.ones(len(bream_length), dtype=torch.int32)
smelt_target = torch.zeros(len(smelt_length), dtype=torch.int32)

fish_target = torch.cat([bream_target, smelt_target])

def sort():
    # sort 함수
    x = torch.randn(10, 2)

    for i in range(len(x)):
        x[i][1] = i

    for i in range(len(x)):
        for j in range(i, len(x)):
            if x[i][0] > x[j][0]:
                temp = x[j]
                x[j] = x[i]
                x[i] = temp

# k-최근접 이웃 알고리즘을 구현해본다.
## 총 데이터를 비교해 거리를 구한다.
### 거리를  sort해서 k만큼 구한다.
#### k개 데이터들이 속하는 클래스 비율을 통해 클래스를 판단한다.
def knn(data_set, target_set, k):
    
    data = torch.cat([data_set, target_set])   #두 데이터를 합친다. 식별할 수 있는 데이터 열이 생긴다.
    
    distance = torch.zeros([len(data_set), 2])   #distance[][0] 은 계산된 거리데이터가, di..[][1]은 순서가, sort해도 기억할 수 있게
    
    # 반복할 떄 자기 자신은 제외하고 계산 해야함, 자기자신을 포함하고 k + 1을 해준다?
    for i in range(len(data)):
        for j in range(len(data)):
            distance_ = 0
            for k in range(2):
                distance_ += data[i][k] - data[j][k]
                
            distance[j], distance[j+1] = distance_, j
            
        # distance에 계산된 거리 데이터가 다 차면 해야하는 것
        ## sort후 k만큼? 더 좋은건 없을까?
        ###sort하면 데이터가 섞일건데 
        ### 섞여도 상관없이 순서 데이터를 넣어줄까?
        ## torch.sort를 사용하려고 했는데 내 목적에 맞지 않는거 같다... 생로 만들어야할 듯

sort()


# plt.scatter(length, weight)
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()
# 강아지 감정 분류하기

## 요약
- 본 프로젝트는 모델 파일이 없으므로, 테스트를 원하시는 분은 sungho.park.826@gmail.com 으로 이메일 보내주세요.
- 요리 레시피 챗봇으로, 총 3가지 종류의 질문을 할 수 있고, 비요리관련 질문에는 일괄적으로 경고메시지로 응답한다.
    1. 요리명 기반 질문:  ex) “오늘 자장면이 먹고 싶네”
    2. 재료명 기반 질문:  ex) “소고기, 양파, 토마토, 마늘이 있는데 이걸로 뭘 먹을까?”
    3. 한식/중식/일식/양식 요리 추천요리:  ex) “한식/중식/일식/양식 중에 아무거나 추천해줘”
    4. 비요리 관련 질문:  ex)  “오늘은 날씨가 어떤가?” → (응답) 요리 관련 질문만 해주세요.
- 분류기에 의해서 질문 타입이 분류되고 해당 타입에 맞는 모델을 불러와 해당 질문에 적절한 응답을 만들어 출력해주는 방식으로 구현
- 가장 중요한 것은 질문 분류기의 성능이었는데, 실제로 테스트를 해보는 코드를 돌려서 확인해본 결과, 약 77% 정확도를 보여줬음.(질문 분류기의 학습시 정확도는 84% 였음)

## 역할
- 본 프로젝트는 총 4명의 팀원으로 구성되었고, 본인의 역할은 사용자로부터 받은 질문을 분석하고 분류하는 ‘질문 타입 분류기’ 모델의 개발을 담당하였음.
- 본인은 챗봇 구현 방안을 결정하고, 여러 모델을 통합하여 테스트해볼 수 있는 챗봇의 최종 통합본을 만드는 역할을 하였음.

## 성과
- 모델을 반복적으로 테스트하고, 테스트를 빠르게 할 수 있는 코드 베이스를 형성하는것도 중요함을 배웠음.
- 전반적인 파이토치 사용법에 대해서 배웠음.
- 역전파를 유보 시켜 gpu메모리가 적은 환경에서도 원하는 배치사이즈로 학습시키는 전략을 배웠음.
- 전반적인 챗봇의 구동 원리를 배웠음.


- Distributed 지원X

# Contrastive Learning for Unpaired Image-to-Image Translation

많은 Unpaired Image-to-Image Translation(e.g. CycleGAN, UNIT, MUNIT)에서는 **Cycle Consistency**를 사용하여 수행함.
 
하지만, **Cycle Consistency**는 Source Domain과 Target Domain 관계 간의 강력한 제약인 bijection을 가정함.
1) 이는 Perfect Reconstruction을 생성하는데 어려움
2) Cycle consistency는 조건부 엔트로피 H(X|Y)의(or H(Y|X)) 상계 -> cycle consistency 최소화 하는 것은 Output y가 Input X를 더욱
의존하도록 함.(Paper: Minimizing Cycle-Consistency loss encourage the output y to be more dependent on input x.)

> **Sol)**
> 
> Patchwise Contrastive Loss를 통해 Cycle consistency를 우회 가능.
> Cntrastive Loss를 사용하면 one-side translation이므로 추가적인 Generator와 Discriminator 필요하지 않음.
> 적은 GPU Memory와 빠른 시간의 학습이 가능
> 
> 
# Architecture

![1](https://user-images.githubusercontent.com/76771847/155045674-16b7a752-d03d-4139-9832-f095b3c3c618.png)

1. Generator Encoder를 통해 Embedding Space으로 맵핑
2. PatchWise Contrastive Loss를 사용함.

**PatwhWise Contrastive**

![2](https://user-images.githubusercontent.com/76771847/155045845-71586009-1162-4b63-9a21-e6fa8d0a449b.png)

1. 이미지 내부에서 여러 patch들을 구함. 
2. Target(Query, 얼룩말 머리)의 Patch와 상응하는 Input Patch 1개와 N개의 Negative Patch간의 Softmax-CrossEntropy

**(논문에 따르면 마지막 끝 Layer에서만 Loss를 구하는 것보다 multi-layer에서 구하는 것이 더 좋은 효과를 가져옴. 또한 이미지 내부의 Patch만 사용하는
것이 외부 이미지를 사용하는 것보다(내부+외부 포함) 더 좋은 효과를 가져옴)**

# Result

| **Source A** | **Fake_B** | **Target B** |
|-------------|------------|--------------|
|![RealA](https://user-images.githubusercontent.com/76771847/155834413-1cb2a17e-8c22-4090-89f2-3f768ecf5e1a.png)| ![fake_B](https://user-images.githubusercontent.com/76771847/155834428-fe91e19e-9e58-4930-af1a-d067dadbec81.png)| ![REALB](https://user-images.githubusercontent.com/76771847/155834431-c0cbca8d-567c-44c3-9d7a-e5adcd8d25d9.png)|


# Implement

> python main.py --gpu 0 --batch-size 1 

*대부분 Default로 설정되어 있음.*

#
# import math as m
#
# #함수정의
# def isPrime(n):
#     prime_yes = True
#
#     nn = int(m.sqrt(n))
#
#     for i in range(2, nn+1):
#         if n % i == 0:
#             prime_yes = False
#             break
#     return prime_yes
#
# while True:
#     n=int(input('Enter an integer number between 2 and 32767: '))
#
#     if n<2 or n>32767:
#         print('Error: You should enter an integer number between 2 and 32767')
#     else :
#         if isPrime(n)==True:
#             print(n,'/ True. It is prime')
#         else :
#             print(n,'/ False. It is not prime')
#         break


# 함수정의
def makeDict(K, V):
    dict = {}
    for k, v in zip(K, V):
        dict[k] = v
    return dict


# Make two tuples K and V and call makeDict to get D
K = ('Korean', 'Mathematics', 'English')
V = (90.3, 85.5, 92.7)
D = makeDict(K, V)

# For every key in K, check if the value obtained from D is correct
for k in K:
    print(k, D[k])

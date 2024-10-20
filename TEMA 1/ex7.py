Probabilitate_A=1/5
Probabilitate_B=1/2
Probabilitate_AandB=1/10

#1.
#A si B sunt independente daca si numai daca P(A intersectat B)=P(A)*P(B)
if Probabilitate_A*Probabilitate_B==Probabilitate_AandB:
    print("A si B sunt independente")
else:
    print("A si B sunt dependente")
#2
#P(A sau B)= P(A)+P(B)-P(A si B)
Probabilitate_de_ratare=Probabilitate_A+Probabilitate_B-Probabilitate_AandB
print(Probabilitate_de_ratare)
#3
#P((A si notB) sau (notA si B))
Probabilitate_sa_rateze_unul=Probabilitate_A*(1-Probabilitate_B)+(1-Probabilitate_A)*Probabilitate_B
print(Probabilitate_sa_rateze_unul)
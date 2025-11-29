import math


size = 4

# création de la grille avec ses états terminaux en 0,0 et last, last
grid = [[0 for _ in range(size)] for _ in range(size)]
grid[0][0] = 1
grid[3][3] = 1

# mes actions posibles
actions = [0,1,2,3]
gama = [1,0.9, 0.8]
epsilon = 0.0001
# toujours -1 selon l'énoncé
def reward_function() -> int :
     return -1

# le prochain état à partir de mon action et de mon current state
def next_s(current_s : int, action : int) : 
    # je récupère un indice en 2D à partir de mon état courant en 1D
    row, col = divmod(current_s, size)
    if action == 0:    # UP
            row -= 1
    elif action == 1:  # RIGHT
        col += 1
    elif action == 2:  # DOWN
        row += 1
    elif action == 3:  # LEFT
        col -= 1
    
    # si on sort on reste en place
    if row < 0 or row >= size or col < 0 or col >= size:
        return current_s 

    # je renvoie un indice en 1D à partir de ma grid en 2D
    return row * size + col

# gama : [1,0.9, 0.8]
# epsilon : 0.0001
# actions : [1,2,3,4]
# next_s : callback de ma fonction qui me donne mon prochain état
# reward_function : callback de ma fonction qui me donne ma récompense (-1)
def value_iteration(gama, epsilon, actions, next_s, reward_function) : 
    value_function = [0 for _ in range(size * size)]
    while (True) : 
        # calcul de la value_function
        stop_crit = 0
        for s in range(size * size) : 
           # états terminaux
           if s == 0 or s== (size*size)-1 : continue

           v_old = value_function[s]
           # je récupère toutes les valeurs pour chacune de mes actions 
           value_function[s] = max([reward_function() + gama * value_function[next_s(s, a)] for a in actions])
           # je prépare mon critère d'arrêt
           stop_crit = max(stop_crit, abs(v_old - value_function[s]))
        if(stop_crit < epsilon) : break

    # après covergence de la value_function je calcule la policy
    policy = [None for _ in range(size * size)]
    for s in range(size * size) :
    # états terminaux
        if s == 0 or s== (size*size)-1 : 
            policy[s] = None
            continue

        current_best_action = None
        # je fixe cette valeur à la plus basse valeur dispo en python
        current_best_value = float("-inf")

        # j'exécute chaque action pour récupérer la meilleure
        for action in actions :
            next_state = next_s(s, action)
            value = -1 + gama * value_function[next_state]

            # je récupère l'action qui me rapporte le plus pour la garder dans ma policy
            if(value > current_best_value) : 
                current_best_value = value
                current_best_action = action

        policy[s] = current_best_action

    return policy, value_function


for g in gama : 
    print("gama : ",  g )
    policy , v_f = value_iteration(g, epsilon=epsilon, next_s=next_s, reward_function=reward_function, actions=actions)
    print("policy : " ,  policy )
    print("value fuction : " , v_f )
              

        
     
    


    


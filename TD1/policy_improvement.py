import math


size = 4

# création de la grille avec ses états terminaux en 0,0 et last, last
grid = [[0 for _ in range(size)] for _ in range(size)]
grid[0][0] = 1
grid[3][3] = 1

# mes actions posibles
actions = [0,1,2,3]
gama = [0.9, 0.8]
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

def argmax(list) -> int :
     maximum = max(list)
     return list.index(maximum)

#je retourne une policy fixe aléatoire en premier lieu 
def random_policy() : 
    p = [actions[0] for _ in range(size*size)]
    p[0] = None
    p[-1] = None
    return p

# return la value function calculée avec la policy actuelle
def policy_eval(policy, g) -> list[float] :
    value_function = [0 for _ in range(size * size)] 
    while (True) : 
        # calcul de la value_function
        stop_crit = 0
        for s in range(size * size) : 
           # états terminaux
           if s == 0 or s== (size*size)-1 : continue

           v_f_old = value_function[s]
           # je récupère ma valeur pour mon action (meme méthode que value_iteration) 
           value_function[s] = reward_function() + g * value_function[next_s(s, policy[s])] 

           # je prépare mon critère d'arrêt
           stop_crit = max(stop_crit, abs(v_f_old - value_function[s]))
        if(stop_crit < epsilon) : break

    return value_function

# je calcule la nouvelle policy en prenant la meilleure value fonction en fonction de mon action
def policy_improvement(value_function, last_policy, g):  
        stable = False
        # je recupère une nouvelle policy prête à être modifiée
        new_policy = random_policy()

        # je mets à jour la nouvelle policy
        for s in range(size*size)  :
             new_policy[s] = argmax([reward_function() + g * value_function[next_s(s,a)] for a in actions])

        # je regarde si il y a convergence
        if last_policy == new_policy  : 
             stable = True

        return stable, new_policy


def policy_iteration(g) : 
    policy = random_policy()
    

    # tant qu'il n'y a pas convergence je continue a modifier ma policy
    while True : 
        value_function = policy_eval(policy, g)
        stable, policy = policy_improvement(value_function, policy, g)
        if stable :
            break

    print("gama", g)
    print("value_function", value_function)
    print("policy", policy)


for g in gama : 
     policy_iteration(g)

    
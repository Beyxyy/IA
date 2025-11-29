import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    This function should update the Q function for a given pair of action-state
    following the q-learning algorithm, it takes as input the Q function, the pair action-state,
    the reward, the next state sprime, alpha the learning rate and gamma the discount factor.
    Return the same input Q but updated for the pair s and a.
    """
    max_next= np.max(Q[sprime]) 
    target= r + gamma * max_next 
    td_error= target- Q[s,a]
    Q[s,a]= Q[s,a] + alpha *td_error
    return Q


def epsilon_greedy(Q, s, epsilone):
    """
    This function implements the epsilon greedy algorithm.
    Takes as unput the Q function for all states, a state s, and epsilon.
    It should return the action to take following the epsilon greedy algorithm.
    """
    # je défini aléatoirement ma prochaine action en fonction de epsilon
    action= None
    if(np.random.rand() < epsilone) : 
       action = np.random.choice(env.action_space.n)
    else:
       action= np.argmax(Q[s])
    return action


if __name__ == "__main__":
    env = gym.make("Taxi-v3")

    env.reset()
    env.render()

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.01 # choose your own

    gamma = 0.8 # choose your own #0.80, 

    epsilon = 0.3 # choose your own

    n_epochs = 2000 # choose your own
    max_itr_per_epoch = 100 # choose your own
    rewards = []

    for e in range(n_epochs):
        r = 0

        S, _ = env.reset()

        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilone=epsilon)

            Sprime, R, done, _, info = env.step(A)

            r += R

            Q = update_q_table(
                Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma
            )

            # Update state and put a stoping criteria

        print("episode #", e, " : r = ", r)
        epsilon = epsilon*0.95 # tentative de baisser epsilone à chaque itération

        rewards.append(r)

    print("Average reward = ", np.mean(rewards))

    # plot the rewards in function of epochs

    print("Training finished.\n")


    """

    Evaluate the q-learning algorihtm

    """

    env.close()
#pemier essai : alpha = 0.01 gama = 0.8 epsilon = 0.3 nb_epoch = 20, Average reward =  -185.5
#deuxieme essaie : alpha = 0.07 gama = 0.9 epsilon = 0.5 nb_epoch = 20 Average reward =  -259.5
#troisieme essaie : alpha = 0.02 gama = 0.85 epsilon = 0.4 nb_epoch = 20 à chaque epsilone greddy je baisse epsilone de 5% Average reward =  -203
#quatrieme essaie : alpha = 0.02 gama = 0.85 epsilon = 0.4 nb_epoch = 25 nb_episode = 1000 à chaque epsilone greddy je baisse epsilone de 5% Average reward =  -1700
#cinquieme essaie : alpha = 0.01 gama = 0.8 epsilon = 0.3 nb_epoch = 25 nb_episode = 1000 à chaque epsilone greddy je baisse epsilone de 5% Average reward =  -1557
#sixieme essaie : alpha = 0.01 gama = 0.90 epsilon = 0.3 nb_epoch = 25 nb_episode = 1000 à chaque epsilone greddy je baisse epsilone de 5% Average reward =  -1551
#sepetieme essaie : alpha = 0.01 gama = 0.90 epsilon = 0.8 nb_epoch = 25 nb_episode = 1000 à chaque epsilone greddy je baisse epsilone de 5% Average reward =  -1570
#8eme essaie : alpha = 0.01 gama = 0.8 epsilon = 0.8 nb_epoch = 25 nb_episode = 1000 à chaque epsilone greddy je baisse epsilone de 5% Average reward =  -1570
#9eme essaie : alpha = 0.01 gama = 0.8 epsilon = 0.8 nb_epoch = 100 nb_episode = 1000 à chaque epsilone greddy je baisse epsilone de 5% Average reward =  -1270 récompense qui converge pour nb epoque et épidose
#10e essaie : alpha = 0.01 gama = 0.8 epsilon = 0.8 nb_epoch = 100 nb_episode = 1000 à chaque epsilone greddy je baisse epsilone de 5% Average reward =  -1270 récompense qui converge pour nb epoque et épidose
#11e essaie : alpha = 0.02 gama = 0.99 epsilon = 0.8 nb_epoch = 100 max_iter = 100 à chaque epsilone greddy je baisse epsilone de 5% Average reward =  -137 récompense qui converge pour nb epoque et épidose
#11e essaie : alpha = 0.01 gama = 0.99 epsilon = 0.8 nb_epoch = 100 max_iter = 100 à chaque epsilone greddy je baisse epsilone de 5% Average reward =  -137 récompense qui converge pour nb epoque et épidose

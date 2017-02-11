from __future__ import print_function

import gym
import tensorflow as tf
import tflearn

tf.global_variables_initializer = tf.initialize_all_variables # hack around csx old tensorflow version


from rlflow.core import tf_utils
from rlflow.policies.f_approx import Network
from rlflow.algos.grad import PolicyGradient

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    def donothing(*args, **kwargs):
        pass
    env.render = donothing

    with tf.Session() as sess:
        # Build neural network
        input_tensor = tflearn.input_data(shape=tf_utils.get_input_tensor_shape(env))
        net = tflearn.fully_connected(input_tensor, 4, activation='sigmoid')
        net = tflearn.fully_connected(net, env.action_space.n, activation='softmax')

        # tell RLFlow that our model is a ANN
        policy = Network(net,
                         sess,
                         Network.TYPE_PG)

        #Setup the policy gradient algorithm in RLflow
        pg = PolicyGradient(env,
                            policy,
                            session=sess,
                            episode_len=1000,
                            discount=True,
                            optimizer='adam')

        #And finally, train for a bit.

        pg.train(max_episodes=50000)
        rewards = pg.test(episodes=10)
        print ("Average: ", float(sum(rewards)) / len(rewards))

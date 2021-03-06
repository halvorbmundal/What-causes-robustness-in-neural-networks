"""
This is the code of the PDG attack from "Towards Deep Learning Models Resistant to Adversarial Attacks"
written by Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipra and Adrian Vladu.

The article can be found on https://arxiv.org/abs/1706.06083
and the code on https://github.com/MadryLab/cifar10_challenge.
"""

import tensorflow as tf
import numpy as np

class LinfPGDAttack:
    def __init__(self, model, epsilon, num_steps, step_size, random_start):
        """Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point."""
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.rand = random_start

        loss = model.xent

        self.grad = tf.gradients(loss, model.x_input)[0]

    def perturb(self, x_nat, y, sess, verbose=True):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
            x = np.clip(x, -0.5, 0.5)  # ensure valid pixel range
        else:
            x = x_nat.astype(np.float)

        for i in range(self.num_steps):
            if i % 100 == 0 and verbose:
                print(f"Step {i} of {self.num_steps} steps", flush=True)
            grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                                  self.model.y_input: y})

            x = np.add(x, self.step_size * np.sign(grad), out=x, casting='unsafe')

            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = np.clip(x, -0.5, 0.5)  # ensure valid pixel range

        return x
import tensorflow as tf
import numpy as np

class MimAttack:
    def __init__(self, model, epsilon, num_steps, random_start=False, momentum=1):
        """Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point."""
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.alpha = epsilon / num_steps
        self.rand = random_start
        self.momentum = momentum

        loss = model.xent

        self.grad = tf.gradients(loss, model.x_input)[0]

    def perturb(self, x_nat, y, sess):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
            x = np.clip(x, -0.5, 0.5)  # ensure valid pixel range
        else:
            x = x_nat.astype(np.float)

        g = 0
        for i in range(self.num_steps):
            if i % 100 == 0:
                print(f"Step {i} of {self.num_steps} steps", flush=True)
            grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                                  self.model.y_input: y})
            normalized_gradients = grad / tf.reduce_mean(tf.abs(grad), [1,2,3], keep_dims=True)
            g = self.momentum * g + normalized_gradients

            x = np.add(x, self.alpha * np.sign(g), out=x, casting='unsafe')

            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = np.clip(x, -0.5, 0.5)  # ensure valid pixel range

        return x
import tensorflow as tf


class Distiller(tf.keras.Model):
    def __init__(self, student, teacher, temperature=3.0, alpha=0.1):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha

    def compile(
            self,
            optimizer='rmsprop',
            loss=None,
            metrics=None,
            loss_weights=None,
            weighted_metrics=None,
            run_eagerly=None,
            steps_per_execution=None,
            jit_compile=None,
            # Our custom arguments must come after standard ones or via kwargs
            **kwargs
    ):
        # Call super with all standard Keras arguments
        super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
            jit_compile=jit_compile
        )

        # Pull your custom distillation losses from the keyword arguments
        self.student_loss_fn = kwargs.get("student_loss_fn")
        self.distill_loss_fn = kwargs.get("distill_loss_fn")

    def train_step(self, data):
        # ... logic from previous steps (update_state, etc.) ...
        x, y = data
        teacher_preds = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            student_preds = self.student(x, training=True)
            student_loss = self.student_loss_fn(y, student_preds)
            distill_loss = self.distill_loss_fn(
                tf.nn.softmax(teacher_preds / self.temperature, axis=-1),
                tf.nn.softmax(student_preds / self.temperature, axis=-1)
            )
            loss = (self.alpha * student_loss) + ((1 - self.alpha) * distill_loss)

        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        self.compiled_metrics.update_state(y, student_preds)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results
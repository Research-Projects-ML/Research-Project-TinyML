import tensorflow as tf


class Distiller(tf.keras.Model):
    def __init__(self, student, teacher, temperature=3.0, alpha=0.1):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha

    def compile(self, optimizer="rmsprop", metrics=None, **kwargs):
        self.student_loss_fn = kwargs.pop("student_loss_fn")
        self.distill_loss_fn = kwargs.pop("distill_loss_fn")

        super().compile(
            optimizer=optimizer,
            metrics=metrics,
        )

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
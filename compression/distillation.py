import numpy as np
import tensorflow as tf
from tensorflow import keras


def kd_loss(teacher_logits, student_logits, true_labels, temperature, alpha):
    """
    Combines two objectives:
    1. Soft target loss: KL divergence between teacher and student softened probability distributions. 
    2. Hard target loss: standard cross-entropy between student predictions and ground truth labels.

    The combined loss is:
        L = alpha * L_soft + (1 - alpha) * L_hard

    Hinton et al. (2015)
    """
    teacher_soft = tf.nn.softmax(teacher_logits / temperature, axis=-1)
    student_soft = tf.nn.log_softmax(student_logits / temperature, axis=-1)

    # KL divergence: sum(teacher_soft * log(teacher_soft / student_soft))
    # Equivalent to: -sum(teacher_soft * student_log_soft) + constant
    soft_loss = -tf.reduce_mean(
        tf.reduce_sum(teacher_soft * student_soft, axis=-1)
    ) * (temperature ** 2)  # Scale by T^2 to restore gradient magnitude

    # Hard targets: standard cross-entropy with ground truth
    hard_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=true_labels,
            logits=student_logits
        )
    )

    return alpha * soft_loss + (1 - alpha) * hard_loss


class KnowledgeDistillationTrainer:
    def __init__(self, teacher, student, config, num_classes):
        self.teacher     = teacher
        self.student     = student
        self.temperature = config['kd_temperature']
        self.alpha       = config['kd_alpha']
        self.num_classes = num_classes
        self.epochs      = config['kd_epochs']
        self.patience    = config.get('kd_patience', 10)

        # Freeze teacher = no gradient updates ever
        self.teacher.trainable = False

        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=config['kd_learning_rate'],
            decay_steps=config['kd_epochs']
        )
        self.optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    def _train_step(self, images, labels):
        # Teacher forward pass = inference mode, no gradient tracking
        teacher_logits = self.teacher(images, training=False)

        with tf.GradientTape() as tape:
            # Student forward pass = training mode
            student_logits = self.student(images, training=True)
            loss = kd_loss(
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                true_labels=labels,
                temperature=self.temperature,
                alpha=self.alpha
            )

        # Compute and apply gradients only for student
        gradients = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.student.trainable_variables)
        )
        return loss

    def _evaluate(self, val_dataset):
        total_correct = 0
        total_samples = 0
        total_loss    = 0.0
        num_batches   = 0

        for images, labels in val_dataset:
            teacher_logits = self.teacher(images, training=False)
            # Accuracy using hard predictions (argmax of student logits)
            student_logits = self.student(images, training=False)

            loss = kd_loss(
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                true_labels=labels,
                temperature=self.temperature,
                alpha=self.alpha
            )

            predictions = tf.argmax(student_logits, axis=-1, output_type=tf.int32)
            total_correct += tf.reduce_sum(
                tf.cast(tf.equal(predictions, tf.cast(labels, tf.int32)), tf.int32)
            ).numpy()
            total_samples += len(labels)
            total_loss    += loss.numpy()
            num_batches   += 1

        return total_loss / num_batches, total_correct / total_samples

    def train(self, train_dataset, val_dataset):
        history = {
            'train_loss': [],
            'val_loss':   [],
            'val_accuracy': []
        }

        best_val_acc     = -np.inf
        best_weights     = None
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training
            train_losses = []
            for images, labels in train_dataset:
                loss = self._train_step(images, labels)
                train_losses.append(loss.numpy())

            epoch_train_loss              = np.mean(train_losses)
            epoch_val_loss, epoch_val_acc = self._evaluate(val_dataset)

            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(epoch_val_loss)
            history['val_accuracy'].append(epoch_val_acc)

            print(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {epoch_train_loss:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f} | "
                f"Val Acc: {epoch_val_acc:.4f}"
            )

            if epoch_val_acc > best_val_acc:
                best_val_acc     = epoch_val_acc
                best_weights     = self.student.get_weights()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}. Best val acc: {best_val_acc:.4f}")
                break

        self.student.set_weights(best_weights)
        return self.student, history


def apply_knowledge_distillation(teacher, student, config, train_dataset, val_dataset, num_classes):
    trainer = KnowledgeDistillationTrainer(
        teacher=teacher,
        student=student,
        config=config,
        num_classes=num_classes
    )
    return trainer.train(train_dataset, val_dataset)
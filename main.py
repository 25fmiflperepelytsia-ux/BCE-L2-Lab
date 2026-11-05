
# Завдання 3. Зміна правильних відповідей

y_true_new = np.array([0, 1, 0, 1])

bce_new = -np.mean(
    y_true_new * np.log(y_pred) +
    (1 - y_true_new) * np.log(1 - y_pred)
)

total_loss_new = bce_new + l2

print("Нова BCE:", round(bce_new, 4))
print("Нова загальна втрата:", round(total_loss_new, 4))
import numpy as np

# Правильні відповіді
y_true = np.array([1, 0, 1, 0])

# Прогнози нейронної мережі
y_pred = np.array([0.9, 0.2, 0.8, 0.1])

# Обчислення BCE
epsilon = 1e-15
y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

bce = -np.mean(
    y_true * np.log(y_pred) +
    (1 - y_true) * np.log(1 - y_pred)
)

print("BCE:", bce)

# L2-регуляризація
weights = np.array([0.5, -0.3, 0.8])
lambda_reg = 0.01

l2 = lambda_reg * np.sum(weights ** 2)

total_loss = bce + l2

print("L2:", l2)
print("Загальна втрата:", total_loss)

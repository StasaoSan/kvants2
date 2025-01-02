import numpy as np
import matplotlib.pyplot as plt

# Физические константы
hbar = 1.0545718e-34       # ℏ — приведённая постоянная Планка, [Дж·с]
m = 9.10938356e-31         # m — масса частицы (электрона), [кг]

# Параметры потенциальной ямы
U0_eV = 50.0               # U0 — глубина потенциальной ямы, [эВ]
U0 = U0_eV * 1.60218e-19   # Перевод U0 из эВ в джоули: [Дж]

a = 1.0e-10                # a — половина ширины ямы, [м] (1 ангстрем)

# Параметры дискретизации
L = 5 * a                   # L — граница области моделирования, [м]
N = 1000                    # N — количество точек сетки
x = np.linspace(-L, L, N)   # x — пространственная сетка, [м]
dx = x[1] - x[0]            # dx — шаг сетки, [м]

# Определение потенциала V(x)
V = np.zeros(N)                    # V(x) — потенциальная энергия, [Дж]
V[np.abs(x) < a] = -U0              # V(x) = -U0 внутри ямы, V(x) = 0 вне ямы

# Построение матрицы Гамильтониана
k_e = -hbar**2 / (2 * m * dx**2)   # коэффициент перед второй производной, [Дж]
diag = np.full(N, -2 * k_e) + V      # основная диагональ: кинетическая + потенциальная энергия
off_diag = np.full(N - 1, k_e)       # вне диагональные элементы: кинетическая энергия

# Построение полной матрицы Гамильтониана
H = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

# Решение задачи собственного значения: H ψ = E ψ
eigenvalues, eigenvectors = np.linalg.eigh(H)

# Преобразование собственных значений энергии из джоулей в электронвольты для удобства
energies_eV = eigenvalues / 1.60218e-19   # [эВ]

# Выбор связанных состояний (энергии меньше нуля)
bound_indices = energies_eV < 0
bound_energies = energies_eV[bound_indices]
bound_states = eigenvectors[:, bound_indices]

# Вывод энергий связанных состояний
num_states_to_display = 3  # количество связанных состояний для вывода

print("Энергии связанных состояний (в эВ):")
for n in range(min(num_states_to_display, len(bound_energies))):
    print(f"Энергия E_{n+1} = {bound_energies[n]:.4f} эВ")

# Нормировка волновых функций
for n in range(bound_states.shape[1]):
    psi = bound_states[:, n]
    # Используем np.trapezoid для численного интегрирования
    norm = np.sqrt(np.trapezoid(np.abs(psi)**2, x))  # Нормировочная константа
    bound_states[:, n] = psi / norm               # Нормировка

# Построение графиков волновых функций
plt.figure(figsize=(10, 6))
for n in range(min(num_states_to_display, len(bound_energies))):
    psi = bound_states[:, n]
    plt.plot(x * 1e10, psi + bound_energies[n], label=f'n={n+1}, E={bound_energies[n]:.2f} эВ')

# Построение потенциала для наглядности
plt.plot(x * 1e10, V / 1.60218e-19, 'k--', label='Потенциал V(x)')

plt.title('Связанные состояния в прямоугольной потенциальной яме')
plt.xlabel('x (Å)')
plt.ylabel('Энергия (эВ) и волновые функции')
plt.legend()
plt.grid(True)
plt.show()

# Вычисление волновых чисел k для связанных состояний
# Формула: k = sqrt(2 * m * (E + U0)) / hbar

# Проверка, что E + U0 > 0 для корректного вычисления sqrt
# E в эВ, U0 уже переведён в эВ
valid_indices = bound_energies + U0_eV > 0

k_values = np.sqrt(2 * m * (bound_energies[valid_indices] * 1.60218e-19 + U0)) / hbar  # [1/м]
k_values_nm = k_values * 1e-9  # перевод в [1/нм]

print("\nВолновые числа k (в 1/нм):")
for n in range(len(k_values_nm)):
    print(f"k_{n+1} = {k_values_nm[n]:.4f} 1/нм")

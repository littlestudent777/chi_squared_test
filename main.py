import numpy as np
from scipy.stats import norm, uniform
import functions as fn

np.random.seed(1)

# Генерация выборок
samples = {
    'Нормальное распределение N=100': norm.rvs(size=100),
    'Нормальное распределение N=20': norm.rvs(size=20),
    'Равномерное распределение N=100': uniform.rvs(size=100),
    'Равномерное распределение N=20': uniform.rvs(size=20)
}

results = {}

# Проведение тестов нормальности для всех выборок
for name, sample in samples.items():
    table, chi2_stat, crit_val, p_val, mu, sigma = fn.chi2_normality_test(sample)

    results[name] = {
        'Таблица': table,
        'χ² статистика': chi2_stat,
        'Критическое значение': crit_val,
        'p-value': p_val,
        'Вывод': "Не отвергаем $H_0$" if chi2_stat < crit_val else "Отвергаем $H_0$",
        'μ (ММП)': mu,
        'σ (ММП)': sigma
    }

# Вывод результатов
fn.print_latex_tables(results)

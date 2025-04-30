import numpy as np
import scipy.stats as stats
from scipy.stats import norm, uniform, chi2
import pandas as pd
from scipy.optimize import minimize

np.random.seed(1)

# Таблица критических значений хи-квадрат
chi2_table = {
    1: 3.841,  # Для alpha=0.05 и df=1
    2: 5.991,  # Для alpha=0.05 и df=2
    3: 7.815,  # Для alpha=0.05 и df=3
    4: 9.488,  # Для alpha=0.05 и df=4
    5: 11.070  # Для alpha=0.05 и df=5
}


def estimate_mle_normal(sample):
    def neg_log_likelihood(params):
        mu, sigma = params
        return -np.sum(norm.logpdf(sample, loc=mu, scale=sigma))

    initial_mu = np.mean(sample)
    initial_sigma = np.std(sample)
    result = minimize(neg_log_likelihood, [initial_mu, initial_sigma],
                      bounds=((None, None), (1e-6, None)))
    return result.x[0], result.x[1]


def chi2_normality_test(sample, alpha=0.05):
    n = len(sample)
    mu, sigma = estimate_mle_normal(sample)
    dist = norm(loc=mu, scale=sigma)

    # Фиксированное количество интервалов
    k = 6

    # Создаем интервалы так, чтобы каждый содержал не менее 5 наблюдений
    # Начинаем с равных вероятностных интервалов
    probs = np.linspace(0, 1, k + 1)
    bins = dist.ppf(probs)

    # Наблюдаемые частоты
    observed, _ = np.histogram(sample, bins=bins)

    # Если есть интервалы с малым числом наблюдений, объединяем с соседними
    i = 0
    while i < len(observed):
        if observed[i] < 5:
            if i == len(observed) - 1 and i > 0:
                # Объединяем с предыдущим интервалом
                observed[i - 1] += observed[i]
                observed = np.delete(observed, i)
                bins = np.delete(bins, i)
            elif i < len(observed) - 1:
                # Объединяем со следующим интервалом
                observed[i] += observed[i + 1]
                observed = np.delete(observed, i + 1)
                bins = np.delete(bins, i + 1)
            else:
                i += 1
        else:
            i += 1

    # Пересчитываем ожидаемые частоты для новых интервалов
    expected_probs = np.diff(dist.cdf(bins))
    expected = n * expected_probs

    # Гарантируем, что df >= 1
    if len(expected) < 3:  # Если осталось меньше 3 интервалов
        # Используем минимальное возможное разбиение (2 интервала)
        bins = np.array([-np.inf, mu, np.inf])
        observed, _ = np.histogram(sample, bins=bins)
        expected_probs = np.diff(dist.cdf(bins))
        expected = n * expected_probs

    # Вычисляем статистику хи-квадрат
    chi2_stat = np.sum((observed - expected) ** 2 / expected)
    df = len(expected) - 1 - 2  # степени свободы

    # Получаем критическое значение из таблицы
    critical_value = chi2_table.get(df, chi2_table[max(chi2_table.keys())])

    p_value = 1 - stats.chi2.cdf(chi2_stat, df) if df > 0 else np.nan

    table = pd.DataFrame({
        'Интервал': [f'({bins[i]:.2f}, {bins[i + 1]:.2f}]' for i in range(len(bins) - 1)],
        'Наблюдаемая частота': observed,
        'Ожидаемая частота': expected,
        '(O-E)²/E': (observed - expected) ** 2 / expected
    })

    return table, chi2_stat, critical_value, p_value, mu, sigma


def print_latex_tables(results):
    for name, res in results.items():
        print(f"\n\\subsection{{Результаты для выборки: {name}}}")
        print(f"Оценки ММП параметров нормального распределения:")
        print(f"$\\hat{{\\mu}} = {res['μ (ММП)']:.3f}$, $\\hat{{\\sigma}} = {res['σ (ММП)']:.3f}$")

        print("\\begin{table}[H]")
        print("\\centering")
        print("\\begin{tabular}{|c|c|c|c|}")
        print("\\hline")
        print("Интервал & Наблюдаемая частота & Ожидаемая частота & $(O-E)^2/E$ \\\\")
        print("\\hline")

        for _, row in res['Таблица'].iterrows():
            print(
                f"{row['Интервал']} & {int(row['Наблюдаемая частота'])} & {row['Ожидаемая частота']:.1f} & {row['(O-E)²/E']:.3f} \\\\")

        print("\\hline")
        print("\\end{tabular}")
        print(f"\\caption{{Таблица частот для проверки нормальности выборки ({name})}}")
        print("\\end{table}")

        print(f"Статистика $\\chi^2$: {res['χ² статистика']:.3f} \\\\")
        print(f"Критическое значение: {res['Критическое значение']:.3f} \\\\")
        print(f"Вывод: {res['Вывод']}")


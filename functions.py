import numpy as np
import scipy.stats as stats


class NormalDistributionTest:
    def __init__(self):
        self.alpha = 0.05

    def generate_sample(self, size, dist_type='normal'):
        """Генерация выборки заданного типа"""
        if dist_type == 'normal':
            return np.random.standard_normal(size=size)
        elif dist_type == 'uniform':
            return np.random.uniform(-np.sqrt(3), np.sqrt(3), size=size)

    def estimate_parameters(self, sample):
        """Оценка параметров методом максимального правдоподобия"""
        mu = np.mean(sample)
        sigma = np.std(sample, ddof=0)  # Для ММП используем несмещенную оценку
        return mu, sigma

    def chi2_test(self, sample, mu, sigma):
        """Критерий согласия хи-квадрат"""
        n = len(sample)
        k = int(1 + 3.3 * np.log10(n))  # Правило Старджесса

        # Границы интервалов для N(mu, sigma)
        percentiles = np.linspace(0, 100, k + 1)[1:-1]
        boundaries = np.percentile(sample, percentiles)
        boundaries = np.concatenate([[-np.inf], boundaries, [np.inf]])

        # Наблюдаемые частоты
        observed, _ = np.histogram(sample, boundaries)

        # Теоретические вероятности
        cdf = stats.norm(loc=mu, scale=sigma).cdf
        prob = np.diff(cdf(boundaries))
        expected = prob * n

        # Статистика хи-квадрат
        chi2 = np.sum((observed - expected) ** 2 / expected)
        critical = stats.chi2.ppf(1 - self.alpha, k - 3)  # k-3 для N(mu,sigma)

        return chi2, critical, observed, prob, expected, boundaries

    def generate_latex_table(self, n, chi2, observed, prob, expected, boundaries,
                             mu, sigma):
        """Генерация LaTeX таблицы с результатами"""
        latex = []
        latex.append("\\begin{table}[H]")
        latex.append("\\centering")
        latex.append("\\begin{tabular}{|c|c|c|c|c|c|c|}")
        latex.append("\\hline")
        latex.append(
            "$i$ & Границы интервалов & $n_i$ & $p_i$ & $np_i$ & $n_i - np_i$ & $\\frac{(n_i - np_i)^2}{np_i}$ \\\\")
        latex.append("\\hline")

        for i in range(len(observed)):
            left = "-\\infty" if boundaries[i] == -np.inf else f"{boundaries[i]:.2f}"
            right = "\\infty" if boundaries[i + 1] == np.inf else f"{boundaries[i + 1]:.2f}"

            diff = observed[i] - expected[i]
            chi_component = diff ** 2 / expected[i]

            row = [
                str(i + 1),
                f"$[{left}, {right}]$",
                str(observed[i]),
                f"{prob[i]:.3f}",
                f"{expected[i]:.1f}",
                f"{diff:.1f}",
                f"{chi_component:.2f}"
            ]
            latex.append(" & ".join(row) + " \\\\")
            latex.append("\\hline")

        latex.append("\\end{tabular}")
        latex.append(
            f"\\caption{{Таблица для $n = {n}$, $\\mu = {mu:.2f}$, $\\sigma = {sigma:.1f}$}}")
        latex.append("\\end{table}")

        return "\n".join(latex)

    def run_tests(self):
        """Основная функция выполнения тестов"""
        sample_size = [20, 100]
        for n in sample_size:
            # Нормальное распределение
            print(f"\n% Нормальное распределение n={n}")
            sample_normal = self.generate_sample(n)
            mu, sigma = self.estimate_parameters(sample_normal)

            chi2, critical, observed, prob, expected, boundaries = self.chi2_test(
                sample_normal, mu, sigma
            )

            print(self.generate_latex_table(
                n, chi2, observed, prob, expected, boundaries, mu, sigma
            ))
            print(f"% χ² наблюдаемое = {chi2:.1f}, критическое = {critical:.1f}")
            print(f"% Гипотеза {'принимается' if chi2 < critical else 'отвергается'}\n")

            # Равномерное распределение
            print(f"\n% Равномерное распределение n={n}")
            sample_uniform = self.generate_sample(n, 'uniform')
            mu_unif, sigma_unif = self.estimate_parameters(sample_uniform)

            chi2_unif, critical_unif, observed_unif, prob_unif, expected_unif, boundaries_unif = self.chi2_test(
                sample_uniform, mu_unif, sigma_unif
            )

            print(self.generate_latex_table(
                n, chi2_unif, observed_unif, prob_unif, expected_unif, boundaries_unif,  mu, sigma
            ))
            print(f"% χ² наблюдаемое = {chi2_unif:.1f}, критическое = {critical_unif:.1f}")
            print(f"% Гипотеза {'принимается' if chi2_unif < critical_unif else 'отвергается'}\n")


if __name__ == "__main__":
    tester = NormalDistributionTest()
    tester.run_tests()

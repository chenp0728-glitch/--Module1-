"""比較高斯波包在實空間與波數空間中的寬度。

波包定義為
    psi(x) = exp[-x^2 / (4 sigma_x^2)] * exp(i k0 x)

因此機率密度 |psi|^2 的標準差為 sigma_x，且理想連續傅立葉
轉換的標準差滿足 sigma_x * sigma_k = 1/2。
"""

import numpy as np
import matplotlib.pyplot as plt


def normalized_density(amplitude: np.ndarray, spacing: float) -> np.ndarray:
    """將振幅的模平方正規化，使其離散積分等於 1。"""
    density = np.abs(amplitude) ** 2
    return density / (np.sum(density) * spacing)


def standard_deviation(axis: np.ndarray, density: np.ndarray, spacing: float) -> float:
    """由已正規化的密度計算標準差。"""
    mean = np.sum(axis * density) * spacing
    variance = np.sum((axis - mean) ** 2 * density) * spacing
    return float(np.sqrt(variance))


def main() -> None:
    # 大範圍與密集取樣可減少週期邊界及離散 FFT 的數值誤差。
    point_count = 2**14
    x_max = 40.0
    x = np.linspace(-x_max, x_max, point_count, endpoint=False)
    dx = x[1] - x[0]

    # k = 2*pi/lambda；k0 控制波包內波紋的疏密。
    k0 = 8.0
    sigma_x_values = [4.0, 2.0, 1.0]

    # np.fft.fftfreq 原本回傳 cycles/length，乘 2*pi 才是角波數 k。
    k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(point_count, d=dx))
    dk = k[1] - k[0]

    fig, axes = plt.subplots(
        len(sigma_x_values), 2, figsize=(12, 9), constrained_layout=True
    )

    print("sigma_x(input)   Delta_x(measured)   Delta_k(measured)   Delta_x*Delta_k")
    print("-" * 79)

    for row, sigma_x in enumerate(sigma_x_values):
        psi = np.exp(-(x**2) / (4 * sigma_x**2)) * np.exp(1j * k0 * x)
        rho_x = normalized_density(psi, dx)

        # dx/sqrt(2*pi) 是傅立葉轉換的尺度因子；正規化後不影響寬度。
        phi = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(psi))) * dx / np.sqrt(2 * np.pi)
        rho_k = normalized_density(phi, dk)

        delta_x = standard_deviation(x, rho_x, dx)
        delta_k = standard_deviation(k, rho_k, dk)
        product = delta_x * delta_k

        print(f"{sigma_x:14.3f}   {delta_x:17.5f}   {delta_k:17.5f}   {product:17.5f}")

        ax_x, ax_k = axes[row]
        ax_x.plot(x, rho_x, color="royalblue", lw=2)
        ax_x.fill_between(x, rho_x, color="royalblue", alpha=0.2)
        ax_x.set_xlim(-12, 12)
        ax_x.set_ylabel(r"probability density")
        ax_x.set_title(rf"Real space: $\Delta x={delta_x:.2f}$")
        ax_x.grid(alpha=0.25)

        ax_k.plot(k, rho_k, color="darkorange", lw=2)
        ax_k.fill_between(k, rho_k, color="darkorange", alpha=0.2)
        ax_k.axvline(k0, color="black", ls="--", lw=1, label=rf"$k_0={k0}$")
        ax_k.set_xlim(k0 - 2.5, k0 + 2.5)
        ax_k.set_title(
            rf"k space: $\Delta k={delta_k:.2f}$, "
            rf"$\Delta x\Delta k={product:.2f}$"
        )
        ax_k.legend(loc="upper right")
        ax_k.grid(alpha=0.25)

    axes[-1, 0].set_xlabel(r"position $x$")
    axes[-1, 1].set_xlabel(r"wave number $k=2\pi/\lambda$")
    fig.suptitle("Gaussian wave packet: narrower in x means wider in k", fontsize=15)

    output_file = "gaussian_wave_packet_fft.png"
    fig.savefig(output_file, dpi=180)
    print(f"\nFigure saved as: {output_file}")
    plt.show()


if __name__ == "__main__":
    main()

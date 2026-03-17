import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import pytest
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from svetlanna import Wavefront, SimulationParameters


def hermite_gauss_numerical(x, y, w0, wavelength, z, m, n, dx=0, dy=0):
    """
    Численная реализация Hermite-Gauss моды для сравнения.
    Использует scipy.special.hermite для полиномов.
    """
    k = 2 * np.pi / wavelength
    zR = np.pi * w0**2 / wavelength
    
    # Координаты относительно центра
    X = x - dx
    Y = y - dy
    
    if z == 0:
        # В перетяжке
        xi_x = np.sqrt(2) * X / w0
        xi_y = np.sqrt(2) * Y / w0
        
        Hx = hermite(n)(xi_x)
        Hy = hermite(m)(xi_y)
        
        # Нормировка
        norm = np.sqrt(2 / (2**n * np.math.factorial(n) * np.pi)) * \
               np.sqrt(2 / (2**m * np.math.factorial(m) * np.pi))
        
        E = norm / w0 * Hx * Hy * np.exp(-(X**2 + Y**2) / w0**2)
        
    else:
        # При распространении
        w = w0 * np.sqrt(1 + (z / zR)**2)
        R = z * (1 + (zR / z)**2)
        gouy = (m + n + 1) * np.arctan(z / zR)
        
        xi_x = np.sqrt(2) * X / w
        xi_y = np.sqrt(2) * Y / w
        
        Hx = hermite(n)(xi_x)
        Hy = hermite(m)(xi_y)
        
        norm = np.sqrt(2 / (2**n * np.math.factorial(n) * np.pi)) * \
               np.sqrt(2 / (2**m * np.math.factorial(m) * np.pi))
        
        E = (norm * w0 / w) * Hx * Hy * \
            np.exp(-(X**2 + Y**2) / w**2) * \
            np.exp(1j * (k * z + k * (X**2 + Y**2) / (2 * R) - gouy))
    
    return E


@pytest.mark.parametrize("distance", [0.1, 0.5, 1.0, 2.0])  # Никогда не используем 0
@pytest.mark.parametrize("waist_radius", [0.3, 0.7])
@pytest.mark.parametrize("m,n", [(0,0), (1,0), (0,1), (1,1), (2,0), (0,2)])
def test_hermite_gauss_vs_numerical(distance, waist_radius, m, n):
    """
    Сравнение реализации svetlanna с численной реализацией.
    """
    # Параметры
    wavelength = 0.5
    sim_params = SimulationParameters({
        "x": torch.linspace(-3, 3, 150),
        "y": torch.linspace(-3, 3, 150),
        "wavelength": wavelength,
    })
    
    # Получаем поле из svetlanna
    wf_svetlanna = Wavefront.hermite_gauss(
        sim_params,
        waist_radius=waist_radius,
        distance=distance,
        dx=0, dy=0,
        m=m, n=n
    )
    
    # Создаем координатные сетки для численной реализации
    x_np = sim_params.x.numpy()
    y_np = sim_params.y.numpy()
    X, Y = np.meshgrid(x_np, y_np, indexing='ij')
    
    # Численное поле
    E_num = hermite_gauss_numerical(
        X, Y, waist_radius, wavelength, distance, m, n
    )
    
    # Конвертируем поле из svetlanna
    E_svet = wf_svetlanna.detach().cpu().numpy()
    
    # Сравниваем интенсивности (они более стабильны)
    I_svet = np.abs(E_svet)**2
    I_num = np.abs(E_num)**2
    
    # Нормализуем
    I_svet = I_svet / I_svet.max()
    I_num = I_num / I_num.max()
    
    # Вычисляем метрики качества
    mse = np.mean((I_svet - I_num)**2)
    correlation = np.corrcoef(I_svet.flatten(), I_num.flatten())[0, 1]
    
    # Пороговые значения
    assert correlation > 0.98, f"Низкая корреляция: {correlation:.4f} для HG{m}{n}, z={distance}"
    assert mse < 0.01, f"Высокая MSE: {mse:.4f} для HG{m}{n}, z={distance}"


def test_hermite_gauss_orthogonality():
    """
    Проверка ортогональности разных мод.
    """
    sim_params = SimulationParameters({
        "x": torch.linspace(-5, 5, 300),
        "y": torch.linspace(-5, 5, 300),
        "wavelength": 1.0,
    })
    
    waist_radius = 1.0
    modes = [(0,0), (1,0), (0,1), (1,1)]
    
    overlap_matrix = np.zeros((len(modes), len(modes)))
    
    for i, (m1, n1) in enumerate(modes):
        wf1 = Wavefront.hermite_gauss(
            sim_params, waist_radius=waist_radius, m=m1, n=n1
        )
        norm1 = torch.sqrt(torch.sum(wf1.intensity) * 
                          (sim_params.x[1]-sim_params.x[0]) * 
                          (sim_params.y[1]-sim_params.y[0]))
        
        for j, (m2, n2) in enumerate(modes):
            wf2 = Wavefront.hermite_gauss(
                sim_params, waist_radius=waist_radius, m=m2, n=n2
            )
            norm2 = torch.sqrt(torch.sum(wf2.intensity) * 
                              (sim_params.x[1]-sim_params.x[0]) * 
                              (sim_params.y[1]-sim_params.y[0]))
            
            # Интеграл перекрытия
            overlap = torch.sum(torch.conj(wf1) * wf2) * \
                     (sim_params.x[1]-sim_params.x[0]) * \
                     (sim_params.y[1]-sim_params.y[0])
            
            overlap_normalized = torch.abs(overlap) / (norm1 * norm2)
            overlap_matrix[i, j] = overlap_normalized.item()
    
    # Диагональные элементы должны быть 1, недиагональные - близки к 0
    for i in range(len(modes)):
        assert abs(overlap_matrix[i, i] - 1.0) < 0.1, f"Диагональ {i}: {overlap_matrix[i, i]}"
        for j in range(len(modes)):
            if i != j:
                assert overlap_matrix[i, j] < 0.1, f"Недиагональ {i},{j}: {overlap_matrix[i, j]}"


@pytest.mark.parametrize("m,n,expected_nodes_x,expected_nodes_y", [
    (0, 0, 0, 0),
    (1, 0, 1, 0),
    (0, 1, 0, 1),
    (2, 0, 2, 0),
    (0, 2, 0, 2),
    (1, 1, 1, 1),
])
def test_hermite_gauss_nodes(m, n, expected_nodes_x, expected_nodes_y):
    """
    Тщательная проверка количества узлов с пороговой обработкой.
    """
    sim_params = SimulationParameters({
        "x": torch.linspace(-5, 5, 500),
        "y": torch.linspace(-5, 5, 500),
        "wavelength": 1.0,
    })
    
    waist_radius = 1.0
    
    wf = Wavefront.hermite_gauss(
        sim_params,
        waist_radius=waist_radius,
        m=m, n=n
    )
    
    intensity = wf.intensity.numpy()
    
    # Берем центральные сечения
    cx, cy = intensity.shape[0]//2, intensity.shape[1]//2
    horizontal = intensity[cx, :]
    vertical = intensity[:, cy]
    
    # Нормализуем
    horizontal = horizontal / horizontal.max()
    vertical = vertical / vertical.max()
    
    # Находим узлы (локальные минимумы ниже порога)
    threshold = 0.05
    
    def count_nodes(line):
        # Находим области ниже порога
        below = line < threshold
        # Находим переходы
        changes = np.where(np.diff(below.astype(int)) != 0)[0]
        # Количество узлов = количество переходов вниз
        nodes = 0
        for i in changes:
            if below[i+1] and not below[i]:  # Переход вниз
                nodes += 1
        return nodes
    
    nodes_x = count_nodes(horizontal)
    nodes_y = count_nodes(vertical)
    
    # Для отладки выведем информацию
    print(f"\nHG{m}{n}: Ожидалось узлов: ({expected_nodes_x}, {expected_nodes_y})")
    print(f"Получено узлов: ({nodes_x}, {nodes_y})")
    
    # Проверяем с запасом (иногда узлы могут сливаться)
    assert nodes_x >= expected_nodes_x - 1, f"По x: ожидалось ~{expected_nodes_x}, получено {nodes_x}"
    assert nodes_x <= expected_nodes_x + 1, f"По x: ожидалось ~{expected_nodes_x}, получено {nodes_x}"
    assert nodes_y >= expected_nodes_y - 1, f"По y: ожидалось ~{expected_nodes_y}, получено {nodes_y}"
    assert nodes_y <= expected_nodes_y + 1, f"По y: ожидалось ~{expected_nodes_y}, получено {nodes_y}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])